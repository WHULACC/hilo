#!/use/bin/env python
#!/usr/bin/env python

"""
Name: trainer.py
"""

import os

import torch

import numpy as np
import torch.nn as nn
# import wandb

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

class MyTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        
        # 用于记录最佳结果
        self.scores = []
        self.lines = []
        self.re_init()

    def train(self):
        best_score, best_iter = 0, -1
        
        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            
            # 训练和评估
            self.train_step()
            score, (res, _) = self.evaluate_step()
            print(f"\n{res}\n")
            
            # 重置统计数据并记录结果
            self.re_init()
            self.add_instance(score, res)

            # 保存最佳模型
            if score > best_score:
                if best_iter > -1:
                    os.remove(self.save_name.format(best_iter))
                best_score, best_iter = score, epoch
                
                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                
                torch.save(
                    {
                        'epoch': epoch,
                        'model': self.model.cpu().state_dict(),
                        'best_score': best_score
                    },
                    self.save_name.format(epoch)
                )
                self.model.to(self.config.device)
            
            # 早停
            elif epoch - best_iter > self.config.patience:
                print(f"Not upgrade for {self.config.patience} steps, early stopping...")
                break
                
        # 最终评估
        score, res = self.final_evaluate(best_iter)
        self.final_score, self.final_res = score, res

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader)
        losses = []

        for data in train_data:
            loss, _ = self.model(**data)
            losses.append(loss.item())
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.config.optimizer.step()
            self.model.zero_grad()

            train_data.set_description(
                f"Epoch {self.global_epoch}, loss:{np.mean(losses):.4f}"
            )

    def evaluate_step(self, dataLoader=None):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        
        for data in dataLoader:
            with torch.no_grad():
                loss, output = self.model(**data)
                self.add_output(data, output)
        
        return self.report_score()

    def final_evaluate(self, epoch=0):
        checkpoint = torch.load(self.save_name.format(epoch), map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        score, res = self.evaluate_step(self.test_loader)
        print(res[0])
        return score, res

    def re_init(self):
        self.preds = defaultdict(list)
        self.golds = defaultdict(list)
        self.keys = ['default']

    def add_instance(self, score, res):
        self.scores.append(score)
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax(self.scores)
        return self.lines[best_id]

    def add_output(self, data, output):
        ecp_predictions, emo_predictions, cause_predictions, masks = output
        predictions = ecp_predictions.argmax(-1).cpu().numpy()
        emo_pred = emo_predictions.argmax(-1).cpu().numpy()
        cause_pred = cause_predictions.argmax(-1).cpu().numpy()
        masks = masks.cpu().numpy()

        for i in range(len(emo_pred)):
            mask = masks[i]
            doc_id = data['doc_ids'][i]
            utt_nums = data['utterance_nums'][i]
            
            # 处理情感预测
            emo_pred_ = emo_pred[i, :utt_nums].tolist()
            emo_gold_ = data['labels'][i, :utt_nums].tolist()
            self.preds['emo'] += emo_pred_
            self.golds['emo'] += emo_gold_
            
            # 处理原因预测
            cause_pred_ = cause_pred[i, :utt_nums].tolist()
            cause_gold_ = data['cause_labels'][i, :utt_nums].tolist()
            self.preds['cause'] += cause_pred_
            self.golds['cause'] += cause_gold_

            # 处理情感-原因对预测
            pair_num = data['pair_nums'][i]
            prediction = predictions[i] * mask
            pred_pairs = np.where(prediction == 1)
            pred_pairs = [(w, z) for w, z in zip(pred_pairs[0], pred_pairs[1]) 
                         if emo_pred_[w] == 1 or cause_pred_[z] == 1]
            pred_pairs = [(doc_id, w, z) for w, z in pred_pairs if w >= z]
            
            self.preds['ecp'] += pred_pairs
            self.golds['ecp'] += [(doc_id, *w) for w in data['pairs'][i][:pair_num].tolist()]

    def report_score(self):
        # 计算ECP分数
        tp = len(set(self.preds['ecp']) & set(self.golds['ecp']))
        fp = len(set(self.preds['ecp']) - set(self.golds['ecp']))
        fn = len(set(self.golds['ecp']) - set(self.preds['ecp']))
        
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0

        # 计算情感和原因分数
        gold_emo = [0 if w == 0 else 1 for w in self.golds['emo']]
        pred_emo = [0 if w == 0 else 1 for w in self.preds['emo']]
        emo = precision_recall_fscore_support(gold_emo, pred_emo, average='binary')
        cause = precision_recall_fscore_support(self.golds['cause'], self.preds['cause'], average='binary')
        
        # 生成结果字符串
        res = (f"Pair Pre. {p*100:.4f}\t Rec. {r*100:.4f}\tF1 {f*100:.4f}\n"
               f"TP {tp}\tPred. {tp+fp}\tGold. {tp+fn}\n"
               f"Emo: Pre. {emo[0]*100:.4f}\t Rec. {emo[1]*100:.4f}\tF1 {emo[2]*100:.4f}\n"
               f"Cause: Pre. {cause[0]*100:.4f}\t Rec. {cause[1]*100:.4f}\tF1 {cause[2]*100:.4f}\n")

        return f, (res, {'p': p, 'r': r, 'default': f, 'emo': emo[2], 'cause': cause[2]})