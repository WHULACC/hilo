#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from src.layer import EnhancedLSTM , Biaffine, RoEmbedding, SupConLoss, MultiHeadAttention, FusionGate, NewFusionGate
from src.tools import set_seed

from transformers.file_utils import ModelOutput
from dataclasses import dataclass
import torch.nn.functional as F

class TextClassification(nn.Module):
    def __init__(self, cfg, tokenizer):
        super(TextClassification, self).__init__()
        self.cfg = cfg 
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)
        self.speaker_embedder = nn.Embedding(len(cfg.speaker_dict), bert_config.hidden_size)
        self.tokenizer = tokenizer

        num_classes = 7 if cfg['emo_cat'] == 'yes' else 2
        num = 2
        self.rope_embedder = RoEmbedding(cfg, bert_config.hidden_size * num)
        # self.fusion = FusionGate(bert_config.hidden_size * num)
        self.fusion = NewFusionGate(bert_config.hidden_size * num)

        drop_rate = 0.1

        self.video_linear = nn.Sequential(
            nn.Linear(cfg.video_dim, bert_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        self.audio_linear = nn.Sequential(
            nn.Linear(cfg.audio_dim, bert_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        self.emotion_linear = nn.Sequential(
            nn.Linear(bert_config.hidden_size * num, cfg.hid_size),
            nn.ReLU(),
            nn.Linear(cfg.hid_size, num_classes)
        )
        self.cause_linear = nn.Linear(bert_config.hidden_size * num, 2)
        self.biaffine = Biaffine(bert_config.hidden_size * num, 2)
        self.contrastive = SupConLoss(temperature=0.1)
        # , bias=(True, False)
        self.dropout = nn.Dropout(cfg['dropout'])

        self.lstm = EnhancedLSTM('drop_connect', bert_config.hidden_size, bert_config.hidden_size, 1, ff_dropout=0.1, recurrent_dropout=0.1, bidirectional=True)

        att_head_size = int(bert_config.hidden_size / bert_config.num_attention_heads)

        self.speaker_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size * 2, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)
        self.reply_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size * 2, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)
        self.global_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size * 2, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)

        self.apply(self._init_esim_weights)

        self.bert = AutoModel.from_pretrained(cfg.bert_path)
    
    def _init_esim_weights(self, module):
        """
        Initialise the weights of the ESIM model.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # if isinstance(module, nn.Linear):
            # nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            # if module.bias is not None:
                # nn.init.zeros_(module.bias)
        # if isinstance(module, nn.Embedding):
            # nn.init.normal_(module.weight, mean=0, std=0.1)
            # nn.init.uniform_(module.weight, -0.1, 0.1)
    
    def get_utt_mask(self, input, utterance_nums, pair_nums, pairs):
        mask = torch.arange(input.shape[1]).unsqueeze(0).to(input.device) < utterance_nums.unsqueeze(-1)
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        triu = torch.flip(torch.flip(torch.triu(torch.ones_like(mask[0])), [1]), [0])
        for i in range(len(utterance_nums)):
            mask[i] = mask[i] * triu
     
        batch_size, seq_len  = input.shape[:2]
        
        gold = input.new_zeros((batch_size, seq_len, seq_len), dtype=torch.long)
        for i in range(len(input)):
            if pair_nums[i] == 0:
                continue
            gold[i, [w[0] for w in pairs[i, :pair_nums[i]]], [w[1] for w in pairs[i, :pair_nums[i]]]] = 1
        return mask, gold
    
    def get_dot_product(self, input, masks, gold_matrix, similarity):
        # input: batch_size, max_utterance_num, hidden_dim
        # utterance_nums: batch_size
        # pairs: batch_size, 2
        product = self.biaffine(input, input).squeeze(-1)
        if len(product.shape) == 3:
            product = product.unsqueeze(-1)
        product = product.transpose(2, 1).transpose(3, 2).contiguous()

        product = product * similarity

        activate_loss = masks.view(-1) == 1
        activate_logits = product.view(-1, 2)[activate_loss]
        activate_gold = gold_matrix.view(-1)[activate_loss]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, self.cfg['loss_weight']]).to(input.device))
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(input.device))
        loss = criterion(activate_logits, activate_gold.long())
        if torch.isnan(loss):
            loss = 0
        return loss, product 
    
    def merge_input(self, input, indices):
        """input: 
        """

        max_utterance_num = max([len(w) for w in indices])

        res = input.new_zeros((len(indices), max_utterance_num, input.shape[-1]))

        for i in range(len(indices)):
            cur_id = indices[i][0][0]
            end_id = indices[i][-1][0]
            cur_lens = 0
            for j in range(cur_id, end_id + 1):
                start = input.new_tensor([w[1] for w in indices[i] if w[0] == j], dtype=torch.long)
                end = input.new_tensor([w[2] - 1 for w in indices[i] if w[0] == j], dtype=torch.long)
                start_rep = torch.gather(input[j], 0, start.unsqueeze(-1).expand(-1, input.shape[-1]))
                end_rep = torch.gather(input[j], 0, end.unsqueeze(-1).expand(-1, input.shape[-1]))

                end = input.new_tensor([w[2] for w in indices[i] if w[0] == j], dtype=torch.long)

                lens = start.shape[0]
                # res[i, cur_lens:cur_lens + lens, :input.shape[-1]] = start_rep
                # res[i, cur_lens:cur_lens + lens, input.shape[-1]:] = end_rep
                res[i, cur_lens:cur_lens + lens] = end_rep + input[j][0].unsqueeze(0)
                cur_lens += lens
        return res
    
    def get_emotion(self, logits, utterance_nums, emotion_labels, emo=True):
        mask = torch.arange(logits.shape[1]).unsqueeze(0).to(logits.device) < utterance_nums.unsqueeze(-1)
        mask = mask.to(logits.device)
        activate_loss = mask.view(-1) == 1
        activate_logits = logits.view(-1, logits.shape[-1])[activate_loss]

        activate_gold = emotion_labels.view(-1)[activate_loss]
        # print(activate_logits.shape)
        if self.cfg.emo_cat == 'yes' and emo:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] + [1.5] * 6).to(logits.device))
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(logits.device))
        loss = criterion(activate_logits, activate_gold.long())
        return loss
    
    def set_mask(self, emotion_logitis, cause_logits, utterance_nums):
        valid_mask = torch.arange(emotion_logitis.shape[1]).unsqueeze(0).to(cause_logits.device) < utterance_nums.unsqueeze(-1)
        valid_mask = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)
        emotion_mask = emotion_logitis.argmax(-1) == 1
        cause_mask = cause_logits.argmax(-1) == 1
        joint_mask = emotion_mask.unsqueeze(-1) | cause_mask.unsqueeze(1)

        mask = valid_mask * joint_mask

        return mask
    
    def align_features(self, feat0, feat1, utterance_nums):
        mask = torch.arange(feat0.shape[1]).unsqueeze(0).to(feat0.device) < utterance_nums.unsqueeze(-1)
        criterion = nn.MSELoss()
        activate_loss = mask.view(-1) == 1
        feat0 = feat0.view(-1, feat0.shape[-1])[activate_loss]
        feat1 = feat1.view(-1, feat1.shape[-1])[activate_loss]
        loss = criterion(feat0, feat1)
        return loss
    
    def do_contrastive(self, text, audio, video, utterance_nums):
        batch_size = text.shape[0]
        losses = []
        for i in range(batch_size):
            cur_text = text[i, :utterance_nums[i]]
            cur_audio = audio[i, :utterance_nums[i]]
            cur_video = video[i, :utterance_nums[i]]
            all_list = torch.cat((cur_text, cur_audio, cur_video), dim=0)
            index = torch.arange(cur_text.shape[0])
            index = torch.cat(([index, index, index]), dim=0)
            mask = index.unsqueeze(0) == index.unsqueeze(-1)
            loss = self.contrastive(all_list, mask=mask)
            losses.append(loss)

        return torch.stack(losses).mean()
    
    def build_hdict(self, input, text, audio, video, utterance_nums):

        res = []
        for i in range(len(utterance_nums)):
            instance = input[i, :utterance_nums[i]]
            res.append(instance)
        res = torch.cat(res, dim=0)

        mm_res = []
        for i in range(len(utterance_nums)):
            instance0 = text[i, :utterance_nums[i]]
            instance1 = audio[i, :utterance_nums[i]]
            instance2 = video[i, :utterance_nums[i]] 
            mm_res.extend([instance0, instance1, instance2])
        mm_res = torch.cat(mm_res, dim=0)
        h_dict = {
            'all': res,
            'sub': mm_res
        }
        return h_dict 
    
        
    def split_sequence(self, input, utterance_nums):
        res = input.new_zeros((len(utterance_nums), max(utterance_nums), input.shape[-1]))
        start = 0
        for i in range(len(utterance_nums)):
            res[i, :utterance_nums[i]] = input[start:start + utterance_nums[i]]
            start += utterance_nums[i]
        return res
    
    def get_kld(self, predtion, speakers, utterance_nums):
        # prediction: batch_size, max_utterance_num, 7
        # kld: batch_size, max_utterance_num
        # first_item: batch_size, max_utterance_num
        # second_item: batch_size, max_utterance_num
        res = []
        # matrix = predtion.new_zeros(predtion.shape[0], predtion.shape[1], predtion.shape[1])
        matrix = torch.eye(predtion.shape[1]).to(predtion.device).unsqueeze(0).repeat(predtion.shape[0], 1, 1)

        for i in range(len(predtion)):
            first_item = predtion[i]
            cur_speaker = speakers[i]
            # print(cur_speaker)
            same_speaker = []
            for j in range(len(cur_speaker)):
                m = 1
                while j - m >= 0 and cur_speaker[j] != cur_speaker[j - m]:
                    m += 1
                same_speaker.append(j - m)
                if m > 1 and j - m != -1:
                    matrix[i, j, j - m: j] = 1
            second_item = predtion[i, same_speaker]

            # 应用softmax函数，使a和b代表概率分布
            a_softmax = F.softmax(first_item, dim=1)
            b_softmax = F.softmax(second_item, dim=1)
            m_x = (a_softmax + b_softmax) / 2

            # kl_div = F.kl_div(a_softmax.log(), b_softmax)
            # kl_div = F.kl_div(b_softmax.log(), a_softmax)
            x = F.kl_div(a_softmax.log(), m_x, reduction='none')
            jsd = 0.5 * F.kl_div(a_softmax.log(), m_x, reduction='none') + 0.5 * F.kl_div(b_softmax.log(), m_x, reduction='none')
            jsd = jsd.sum(-1)
            res.append(jsd)
        res = torch.stack(res)
        res = res.unsqueeze(-1) * matrix

        similarity = res.unsqueeze(-1).repeat(1, 1, 1, 2)
        similarity[..., 0] = 0
        similarity = torch.exp(similarity * self.cfg.alpha)
        # for w in res:
            # for kk in w:
                # print([round(w * 100, 2) for w in kk.tolist()])
        return similarity 
    
    def build_attention(self, sequence_outputs, gmasks=None, smasks=None, rmasks=None):
        """
        sequence_outputs: batch_size, seq_len, hidden_size
        speaker_matrix: batch_size, num, num 
        head_matrix: batch_size, num, num 
        """
        # speaker_masks = smasks.bool().unsqueeze(1)
        # reply_masks = rmasks.bool().unsqueeze(1)
        # global_masks = gmasks.bool().unsqueeze(1)


        rep = self.reply_attention(sequence_outputs, sequence_outputs, sequence_outputs, rmasks)[0]
        thr = self.global_attention(sequence_outputs, sequence_outputs, sequence_outputs, gmasks)[0]
        sp = self.speaker_attention(sequence_outputs, sequence_outputs, sequence_outputs, smasks)[0]
        r = torch.stack((rep, thr, sp), 0)
        r = torch.max(r, 0)[0]

        length = sequence_outputs.shape[1] // 4

        return r[:, : length]


    def forward(self, **kwargs):
    # def forward(self, input_ids, input_masks, utterance_nums, pairs, pair_nums, labels, indices):
        input_ids, input_masks, utterance_nums = [kwargs[w] for w in 'input_ids input_masks utterance_nums'.split()]
        pairs, pair_nums, labels, indices = [kwargs[w] for w in 'pairs pair_nums labels indices'.split()]
        cause_labels, speaker_ids = [kwargs[w] for w in ['cause_labels', 'speaker_ids']]
        audio_features, video_features = [kwargs[w] for w in ['audio_features', 'video_features']]
        # hgraphs = kwargs['hgraph']
        gmasks, smasks, rmasks = [kwargs[w] for w in ['gmasks', 'smasks', 'rmasks']]

        input = self.bert(input_ids, attention_mask=input_masks)[0]
        # input = self.dropout(input)

        speaker_emb = self.speaker_embedder(speaker_ids)

        text = self.merge_input(input, indices)
        audio = self.audio_linear(audio_features)
        video = self.video_linear(video_features)

        input = text + speaker_emb + audio + video

        input = self.lstm(input, None, utterance_nums.cpu())

        tt = torch.cat((text, text), dim=-1)
        aa = torch.cat((audio, audio), dim=-1)
        vv = torch.cat((video, video), dim=-1)

        sequence_input = torch.cat((input, tt, aa, vv), 1)

        output = self.build_attention(sequence_input, gmasks, smasks, rmasks)
        # input = input + output
        # input = output
        # input = torch.cat((input, output), dim=-1)
        input = self.fusion(input, output)

        emotion_logits = self.emotion_linear(input)
        emo_loss = self.get_emotion(emotion_logits, utterance_nums, labels, emo=True)

        cause_logits = self.cause_linear(input)
        cause_loss = self.get_emotion(cause_logits, utterance_nums, cause_labels, emo=False)

        ecp_mask, gold_matrix = self.get_utt_mask(input, utterance_nums, pair_nums, pairs)

        # joint_mask = self.set_mask(emotion_logits, cause_logits, utterance_nums)
        similarity = self.get_kld(emotion_logits, speaker_ids, utterance_nums)

        ecp_loss, ecp_logits = self.get_dot_product(input, ecp_mask, gold_matrix, similarity)
        rop_loss, rop_logits = self.rope_embedder.classify_matrix(input, gold_matrix, ecp_mask, similarity)

        loss = rop_loss + emo_loss + cause_loss + ecp_loss
        # loss = rop_loss + emo_loss + cause_loss
        # loss = emo_loss + cause_loss + ecp_loss 

        return loss, (rop_logits + ecp_logits, emotion_logits, cause_logits, ecp_mask)
        # return loss, (rop_logits, emotion_logits, cause_logits, ecp_mask)