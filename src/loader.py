#!/usr/bin/env python

"""
Name: loader.py
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import logging
import pickle as pkl
from typing import Dict
from dataclasses import dataclass 
from collections import Counter
import numpy as np

from src.layer import build_hgraph



def build_mask(utterance_nums, speakers):
    max_utterance = max(utterance_nums)

    gmask = torch.zeros(len(utterance_nums), max_utterance, max_utterance, dtype=torch.long)
    for i in range(len(utterance_nums)):
        gmask[i, :utterance_nums[i], :utterance_nums[i]] = 1
    gmask = gmask.repeat(1, 4, 4)

    smask = torch.zeros(len(utterance_nums), max_utterance, max_utterance, dtype=torch.long)
    for i in range(len(speakers)):
        speaker = speakers[i]
        m = np.array([[1 if i == j else 0 for i in speaker] for j in speaker])
        smask[i, :utterance_nums[i], :utterance_nums[i]] = torch.tensor(m)
    smask = smask.repeat(1, 4, 4)

    rmaks = torch.zeros(len(utterance_nums), max_utterance, max_utterance, dtype=torch.long)
    for i in range(len(utterance_nums)):
        utterance_num = utterance_nums[i]
        eye = np.eye(utterance_num) + np.eye(utterance_num, k=1) + np.eye(utterance_num, k=-1)
        # eye = torch.eye(utterance_num) + torch.eye(utterance_num, offset=1) + torch.eye(utterance_num, offset=-1)
        rmaks[i, :utterance_nums[i], :utterance_nums[i]] = torch.tensor(eye, dtype=torch.long)
    rmaks = rmaks.repeat(1, 4, 4)
    return gmask, smask, rmaks

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data, mode):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.data = data[mode]
        self.label_dict = data['label_dict']
        self.speaker_dict = data['speaker_dict']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        keys = list(self.data[i])
        values = [self.data[i][k] for k in keys]
        values[4] = [self.label_dict[w] for w in values[4]]
        values[3] = [self.speaker_dict[w] for w in values[3]]
        return (keys, values)

@dataclass
class CollateFN:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    config: Dict
    video_map: Dict 
    audio_map: Dict 
    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        keys, instances = list(zip(*instances))
        doc_ids = [line[0] for line in instances]
        pairs = [[(w - 1, z - 1) for w, z in line[1]] for line in instances]
        utterances = [line[2] for line in instances]
        emotions = [line[4] for line in instances]
        speakers = [line[3] for line in instances]
        label_dict = self.config['label_dict']

        if self.config.emo_cat != 'yes':
            emotions = [[0 if w == label_dict['neutral'] else 1 for w in line] for line in emotions]

        utterance_nums = [len(line) for line in utterances]
        gmasks, smasks, rmasks = build_mask(utterance_nums, speakers)
        IGNORE_INDEX = -100
        # max_utterance = max(max(utterance_nums), 2)
        max_utterance = max(utterance_nums)

        emotions = [w + [IGNORE_INDEX] * (max_utterance - len(w)) for w in emotions]
        res = []
        max_length = self.config['max_length'] 
        total_length = self.config['total_length']
        # padd_utterance = [w + [''] * (max_utterance - len(w)) for w in utterances]
        # for i in range(len(utterances)):
            # batch_input = self.tokenizer.batch_encode_plus(padd_utterance[i], return_tensors="pt", pad_to_max_length=True, max_length=max_length)
            # res.append(batch_input)
        input_tokens, indices = pack(utterances, max_length, total_length, self.tokenizer, self.config)

        max_seq_len = max([len(w) for w in input_tokens])
        input_tokens = [w + [self.config.pad] * (max_seq_len - len(w)) for w in input_tokens]

        input_ids = [self.tokenizer.convert_tokens_to_ids(w) for w in input_tokens]
        # input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [[1] * len(w) + [0] * (max_seq_len - len(w)) for w in input_tokens]

        # input_ids = torch.stack([w['input_ids'] for w in res], dim=0)
        # attention_mask = torch.stack([w['attention_mask'] for w in res], dim=0)
        
        # padding pairs
        pair_nums = [len(line) for line in pairs]
        max_pair = max(pair_nums)
        pairs = [w + [(IGNORE_INDEX, IGNORE_INDEX)] * (max_pair - len(w)) for w in pairs]

        cuase_labels = [[0 for _ in range(max_utterance)] for _ in range(len(pairs))]
        # emotion_binary = [[0 for _ in range(max_utterance)] for _ in range(len(pairs))]
        for i in range(len(pairs)):
            for w, z in pairs[i]:
                if z != IGNORE_INDEX:
                    cuase_labels[i][z] = 1
                # if w != IGNORE_INDEX:
                    # emotion_binary[i][w] = 1
        
        speakers = [w + [0] * (max_utterance - len(w)) for w in speakers]

        video_features = []
        for i in range(len(doc_ids)):
            video_feature = np.stack([self.video_map[(doc_ids[i], j)] for j in range(1, utterance_nums[i] + 1)])
            video_features.append(video_feature)
        video_features = np.stack([np.concatenate([w, np.zeros((max_utterance - w.shape[0], w.shape[1]))], axis=0) for w in video_features])
        # limit value to -1 to 1
        video_features = np.clip(video_features, -1, 1)

        audio_features = []
        for i in range(len(doc_ids)):
            audio_feature = np.stack([self.audio_map[(doc_ids[i], j)]  for j in range(1, utterance_nums[i] + 1)])
            audio_features.append(audio_feature)
        audio_features = np.stack([np.concatenate([w, np.zeros((max_utterance - w.shape[0], w.shape[1]))], axis=0) for w in audio_features])
        # limit value to -1 to 1
        audio_features = np.clip(audio_features, -1, 1)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long).to(self.config['device']),
            'input_masks': torch.tensor(attention_mask, dtype=torch.long).to(self.config['device']),
            'indices': indices, # 'indices': [(global_id, start, end), ...]
            'utterance_nums': torch.tensor(utterance_nums, dtype=torch.long).to(self.config['device']),
            'pairs': torch.tensor(pairs, dtype=torch.long).to(self.config['device']),
            'pair_nums': torch.tensor(pair_nums, dtype=torch.long).to(self.config['device']),
            'labels': torch.tensor(emotions, dtype=torch.long).to(self.config['device']),
            'cause_labels': torch.tensor(cuase_labels, dtype=torch.long).to(self.config['device']),
            'doc_ids': doc_ids,
            'speaker_ids': torch.tensor(speakers, dtype=torch.long).to(self.config['device']),
            'video_features': torch.tensor(video_features, dtype=torch.float).to(self.config['device']),
            'audio_features': torch.tensor(audio_features, dtype=torch.float).to(self.config['device']),
            'gmasks': gmasks.to(self.config['device']),
            'smasks': smasks.to(self.config['device']),
            'rmasks': rmasks.to(self.config['device']), 
            # 'hgraph': hgraphs,
        }

def pack(dialogues, max_len, total_len, tokenizer, config):
    res = []
    indices = []
    for i in range(len(dialogues)):
        cur_res = [config.cls]
        cur_indices = []
        for line in dialogues[i]:
            tokens = tokenizer.tokenize(line)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            if len(cur_res) + len(tokens) > total_len:
                res.append(cur_res)
                cur_res = [config.cls]
            global_id = len(res) 
            start = len(cur_res)
            cur_res += tokens + [config.sep]
            end = len(cur_res) - 1
            cur_indices.append((global_id, start, end))
        res.append(cur_res)
        indices.append(cur_indices)
    # res = [tokenizer.convert_tokens_to_ids(w) for w in res]
    # input_masks = [[1] * len(w) for w in res]
    return res, indices 

def read_data(path):
    with open(path, 'r') as f:
        data = f.read().splitlines()

    structured_data = []
    idx = 0
    while idx < len(data):
        line = data[idx]
        scene_id, num_lines = map(int, line.split(' '))
        if len(data[idx+1].strip()) > 0:
            emotion_cause_pairs = [tuple(map(int, pair.split(','))) for pair in data[idx + 1].strip('()').split('),(')]
        else:
            emotion_cause_pairs = []
        lines, timecodes, speakers, emotions = [], [], [], []
        for i in range(num_lines):
            line_parts = data[idx + 2 + i].split(' | ')
            utterance_id, speaker, emotion, utterance = line_parts[:4]
            timecode = line_parts[4]
            speakers.append(speaker)
            emotions.append(emotion)
            timecodes.append(timecode)
            lines.append(utterance)

        structured_data.append({'doc_id': scene_id, 'emotion_cause_pairs': emotion_cause_pairs, 'lines': lines, 'speakers': speakers, 'emotions': emotions, 'timecodes': timecodes})
        idx += 2 + num_lines
    return structured_data

def read_video(video=True):
    if video:
        path = './data/dataset/video_embedding_4096.npy'
    else:
        path = './data/dataset/audio_embedding_6373.npy'
    map_path = './data/dataset/video_id_mapping.npy'
    video_feature = np.load(path)
    id_map = np.load(map_path, allow_pickle=True).item()
    get_num = lambda x: (int(x.split('utt')[0][3:]), int(x.split('utt')[1]))
    id_map = {get_num(w): video_feature[z] for w, z in id_map.items()}
    return id_map

def build_dict(data):
    wordlist = []
    for line in data:
        wordlist.extend(line['emotions'])
    wordcount = Counter(wordlist)
    word2dict = {w: i for i, w in enumerate(wordcount.keys())}
    return word2dict 

def build_speaker_dict(data):
    wordlist = []
    for line in data:
        wordlist.extend(line['speakers'])
    wordcount = Counter(wordlist)
    word2dict = {w: i for i, w in enumerate(wordcount.keys())}
    return word2dict

def make_supervised_data_module(config, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    path = os.path.join(config.preprocessed_dir, '{}.pkl'.format(config.bert_path))
    if not os.path.exists(path):
        data = {}
        for mode in ['train', 'valid', 'test']:
            data[mode] = read_data(os.path.join(config.dataset_dir, '{}.txt'.format(mode)))
        
        data['video'] = read_video()
        data['audio'] = read_video(False)
        data['label_dict'] = build_dict(data['train'])
        data['speaker_dict'] = build_speaker_dict(data['train'] + data['valid'] + data['test'])
        with open(path, 'wb') as f:
            pkl.dump(data, f)
    else:
        with open(path, 'rb') as f:
            data = pkl.load(f)
    config['label_dict'] = data['label_dict']
    config['speaker_dict'] = data['speaker_dict']
    train_dataset = SupervisedDataset(data, 'train')
    valid_dataset = SupervisedDataset(data, 'valid')
    test_dataset = SupervisedDataset(data, 'test')
    if config.model_name == 'bert':
        data_collator = CollateFN(tokenizer=tokenizer, config=config, video_map=data['video'], audio_map=data['audio'])
    elif config.model_name == 'lstm':
        data_collator = CollateFNLSTM(tokenizer=tokenizer, word_dict=data['word_dict'])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=data_collator)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=data_collator)

    return train_loader, valid_loader, test_loader, config
    return dict(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, config=config)