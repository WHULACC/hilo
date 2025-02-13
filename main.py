#!/usr/bin/env python

import os
import argparse
import yaml
import random
import string
import torch
from attrdict import AttrDict
from loguru import logger
import warnings

from src.tools import update_config, set_seed, load_params_bert
from src.trainer import MyTrainer 
from src.loader import make_supervised_data_module
import transformers
from src.model import TextClassification

warnings.filterwarnings('ignore')

class Template:
    def __init__(self, args):
        # 加载配置文件
        config = AttrDict(yaml.load(
            open('src/config.yaml', 'r', encoding='utf-8'), 
            Loader=yaml.FullLoader
        ))
        
        # 更新配置
        for k, v in vars(args).items():
            setattr(config, k, v)
        config = update_config(config)
        
        # 设置模型保存名称
        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        config.save_name = f"{config.model_name}_{random_str}_{config.seed}_{{}}.pt"
        
        # 设置随机种子和设备
        set_seed(config.seed)
        config.device = torch.device(f'cuda:{config.cuda_index}' if torch.cuda.is_available() else 'cpu')
        
        self.config = config

    def forward(self):
        # 初始化tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.bert_path, 
            padding_side="right",
            use_fast=False
        )

        # 准备数据
        self.train_loader, self.valid_loader, self.test_loader, self.config = \
            make_supervised_data_module(self.config, tokenizer)

        # 初始化模型
        if self.config.model_name == 'bert':
            self.model = TextClassification(self.config, tokenizer).to(self.config.device)

        # 加载优化器等参数
        self.config = load_params_bert(self.config, self.model, self.train_loader) 

        # 训练模型
        trainer = MyTrainer(self.model, self.config, self.train_loader, self.valid_loader, self.test_loader)
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert', help='model type')
    parser.add_argument('-cd', '--cuda_index', type=int, default=0, help='cuda device index')
    
    template = Template(parser.parse_args())
    template.forward()
