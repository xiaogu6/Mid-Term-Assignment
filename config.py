import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    d_model: int = 128
    n_head: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 1024
    vocab_size: int = 65  # Tiny Shakespeare字符数
    
@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 20
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.98)
    weight_decay: float = 0.01
    
@dataclass
class DataConfig:
    dataset_name: str = "tiny_shakespeare"
    train_split: float = 0.9
    seq_len: int = 256

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
