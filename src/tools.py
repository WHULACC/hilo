#!/use/bin/env python


import os
import random

import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

def update_config(config):

    dirs = ['preprocessed_dir', 'target_dir', 'dataset_dir']
    for dirname in dirs:
        if dirname in config:
            config[dirname] = os.path.join(config.data_dir, config[dirname])
    config['emb_file'] = os.path.join(config.preprocessed_dir, config['emb_file'])
    if not os.path.exists(config.preprocessed_dir):
        os.makedirs(config.preprocessed_dir)
    if not os.path.exists(config.target_dir):
        os.makedirs(config.target_dir)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    pairs = predictions.pairs
    pairs = predictions.pair_nums
    pairs = predictions['pairs']
    pair_nums = predictions['pair_nums']
    logits = predictions['logits']
    # The report includes precision, recall, f1-score and support (number of instances) for each class.
    report = classification_report(labels, predictions, output_dict=True)
    for class_id, metrics in report.items():
        if class_id.isdigit():
            print(f'Class {class_id}: F1 Score: {metrics["f1-score"]}')
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': precision_recall_fscore_support(labels, predictions, average='weighted')[2],
    }
    

def load_params_bert(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight']
    bert_params = set(model.bert.parameters())
    other_params = list(set(model.parameters()) - bert_params)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': float(config.bert_lr), 'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'lr': float(config.bert_lr), 'weight_decay': 0.0},
        {'params': other_params, 'lr': float(config.learning_rate), 'weight_decay': float(config.weight_decay)},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.epoch_size * fold_data.__len__())

    config.optimizer = optimizer
    config.scheduler = scheduler

    return config