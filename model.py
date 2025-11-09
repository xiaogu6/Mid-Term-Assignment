import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len = Q.size(0), Q.size(1)
        
        residual = Q
        
        # Linear projections
        Q = self.W_Q(Q).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        # Apply attention
        context, attn_weights = self.attention(Q, K, V, mask=mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.W_O(context)
        output = self.dropout(output)
        
        # Add & Norm
        output = self.layer_norm(output + residual)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        output = self.linear1(x)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        x, attn = self.self_attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x, self_attn = self.self_attention(x, x, x, tgt_mask)
        x, encoder_attn = self.encoder_attention(x, encoder_output, encoder_output, src_mask)
        x = self.feed_forward(x)
        return x, self_attn, encoder_attn

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, num_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)
            
        x = self.layer_norm(x)
        return x, attentions

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, num_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocgab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        self_attentions = []
        encoder_attentions = []
        
        for layer in self.layers:
            x, self_attn, encoder_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attentions.append(self_attn)
            encoder_attentions.append(encoder_attn)
            
        x = self.layer_norm(x)
        return x, self_attentions, encoder_attentions

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            config.vocab_size, config.d_model, config.n_head, 
            config.d_ff, config.num_encoder_layers, config.dropout, config.max_seq_len
        )
        self.decoder = Decoder(
            config.vocab_size, config.d_model, config.n_head, 
            config.d_ff, config.num_decoder_layers, config.dropout, config.max_seq_len
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output, encoder_attentions = self.encoder(src, src_mask)
        decoder_output, self_attentions, encoder_decoder_attentions = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        output = self.output_projection(decoder_output)
        return output, encoder_attentions, self_attentions, encoder_decoder_attentions
    
    def generate(self, src, max_len, temperature=1.0):
        self.eval()
        with torch.no_grad():
            encoder_output, _ = self.encoder(src)
            
            # 开始生成（这里简化处理，实际应该用更复杂的生成策略）
            generated = torch.tensor([[1]], device=src.device)  # 假设1是开始符
            
            for _ in range(max_len - 1):
                tgt_mask = self._generate_square_subsequent_mask(generated.size(1))
                decoder_output, _, _ = self.decoder(generated, encoder_output, tgt_mask=tgt_mask)
                next_token_logits = self.output_projection(decoder_output[:, -1, :]) / temperature
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
            return generated
            
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
