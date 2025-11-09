import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.train_losses = []
        self.val_losses = []
        self.val_ppls = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # 为语言建模任务创建输入和目标
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].contiguous().view(-1)
            
            # 创建因果掩码
            seq_len = tgt_input.size(1)
            tgt_mask = self.model._generate_square_subsequent_mask(seq_len).to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            output, _, _, _ = self.model(src, tgt_input, tgt_mask=tgt_mask)
            output = output.view(-1, output.size(-1))
            
            loss = self.criterion(output, tgt_output)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
                
        return total_loss / total_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_batches = 0
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:].contiguous().view(-1)
                
                seq_len = tgt_input.size(1)
                tgt_mask = self.model._generate_square_subsequent_mask(seq_len).to(self.device)
                
                output, _, _, _ = self.model(src, tgt_input, tgt_mask=tgt_mask)
                output = output.view(-1, output.size(-1))
                
                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()
                total_batches += 1
                
        avg_loss = total_loss / total_batches
        perplexity = np.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def train(self):
        print("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss, perplexity = self.validate()
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ppls.append(perplexity)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{self.config.num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Perplexity: {perplexity:.2f}')
            print(f'  Time: {epoch_time:.2f}s')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
                
        self.plot_training_curve()
        
    def save_checkpoint(self, epoch, is_best=False):
        Path('checkpoints').mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_ppl': self.val_ppls
        }
        
        if is_best:
            torch.save(checkpoint, 'checkpoints/best_model.pth')
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pth')
        
    def plot_training_curve(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_ppls)
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Validation Perplexity')
        
        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
