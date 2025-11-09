#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
from src.model import Transformer
from src.data_loader import download_tiny_shakespeare, TinyShakespeareDataset
from src.config import ModelConfig
from src.utils import set_seed, get_device, generate_text

def main():
    parser = argparse.ArgumentParser(description='Generate text with trained Transformer')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--prompt', type=str, default='First Citizen:', help='Text prompt for generation')
    parser.add_argument('--max_len', type=int, default=500, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    # 加载数据以获取词汇表
    data = download_tiny_shakespeare()
    dataset = TinyShakespeareDataset(data, seq_len=256, split='train')
    
    # 创建模型配置
    config = ModelConfig(vocab_size=dataset.vocab_size)
    
    # 创建模型并加载权重
    model = Transformer(config)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 生成
