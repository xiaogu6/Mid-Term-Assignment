import torch
from torch.utils.data import Dataset, DataLoader
import requests
from pathlib import Path

class TinyShakespeareDataset(Dataset):
    def __init__(self, data, seq_len, split='train', train_ratio=0.9):
        self.seq_len = seq_len
        self.data = data
        self.vocab = sorted(list(set(data)))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        
        # 分割数据集
        split_idx = int(len(data) * train_ratio)
        if split == 'train':
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]
            
        self.tokenized_data = [self.char_to_idx[ch] for ch in self.data]
        
    def __len__(self):
        return len(self.tokenized_data) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.tokenized_data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokenized_data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

def download_tiny_shakespeare(data_dir='data'):
    Path(data_dir).mkdir(exist_ok=True)
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    file_path = Path(data_dir) / 'tinyshakespeare.txt'
    if not file_path.exists():
        print("Downloading Tiny Shakespeare dataset...")
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_data_loaders(seq_len=256, batch_size=64, train_ratio=0.9):
    data = download_tiny_shakespeare()
    dataset = TinyShakespeareDataset(data, seq_len, 'train', train_ratio)
    val_dataset = TinyShakespeareDataset(data, seq_len, 'val', train_ratio)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.vocab_size, dataset.char_to_idx, dataset.idx_to_char
