import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_text(model, char_to_idx, idx_to_char, prompt, max_len=100, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # 将提示符转换为索引
        input_indices = [char_to_idx[ch] for ch in prompt]
        input_tensor = torch.tensor([input_indices], device=model.device)
        
        generated = input_tensor.clone()
        
        for _ in range(max_len):
            # 创建因果掩码
            seq_len = generated.size(1)
            tgt_mask = model._generate_square_subsequent_mask(seq_len).to(model.device)
            
            # 前向传播
            output, _, _, _ = model(generated, generated, tgt_mask=tgt_mask)
            next_token_logits = output[:, -1, :] / temperature
            
            # 应用softmax获取概率
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # 从分布中采样
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # 如果生成了结束符，提前停止
            if next_token.item() == char_to_idx.get('\n', -1):
                break
                
        # 将索引转换回文本
        generated_text = ''.join([idx_to_char[idx.item()] for idx in generated[0]])
        return generated_text
