#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
from src.model import Transformer
from src.data_loader import get_data_loaders
from src.trainer import TransformerTrainer
from src.config import ModelConfig, TrainingConfig, DataConfig
from src.utils import set_seed, get_device

def main():
    parser = argparse.ArgumentParser(description='Train Transformer model')
    parser.add_argument('--config', type=str, default='../configs/base.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    # 配置参数
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # 获取数据加载器
    train_loader, val_loader, vocab_size, char_to_idx, idx_to_char = get_data_loaders(
        seq_len=data_config.seq_len,
        batch_size=training_config.batch_size
    )
    
    # 更新词汇表大小
    model_config.vocab_size = vocab_size
    
    # 创建模型
    model = Transformer(model_config)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 创建训练器
    trainer = TransformerTrainer(model, train_loader, val_loader, training_config, device)
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()
