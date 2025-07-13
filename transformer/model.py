import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def position_embedding(seq_length, embed_dim):
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
    pe = torch.zeros(seq_length, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # 适应批次输入
    return pe


class SignalEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, seq_length=2048):
        super(SignalEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = seq_length // patch_size

        self.proj = nn.Linear(patch_size, embed_dim)  # 线性投影层
        pos_embed = position_embedding(self.num_patches, embed_dim)  # 计算正弦位置编码
        self.register_buffer('pos_embed', pos_embed)  # 注册为buffer，不会被优化器更新
    
    def forward(self, x):
        # print('Signal embedding (in):', x.shape)
        # x: (batch_size, window_size)
        patches = x.unfold(1, self.patch_size, self.patch_size)
        x = self.proj(patches)  # (batch_size, num_patches, embed_dim)
        x += self.pos_embed
        # print('Signal embedding (out):', x.shape)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # print('Transformer block (in):', x.shape)
        attn_output, _ = self.self_attn(x, x, x)  # Self-attention
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        # print('Transformer block (out):', x.shape)
        return x


class DiagnosticsModel(nn.Module):
    def __init__(self, num_classes, patch_size=16, embed_dim=128, num_heads=8, num_blocks=6, expansion=4):
        super(DiagnosticsModel, self).__init__()
        self.signal_embedding = SignalEmbedding(patch_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),  # 扩展维度
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * expansion, num_classes),
        )

    def forward(self, x):
        # print('Start:', x.shape)
        x = self.signal_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        # print('End:', x.shape)
        cls_tokens = x.mean(dim=1)
        # print('CLS token:', cls_tokens.shape)
        class_pred = self.classifier(cls_tokens)
        # print('Output:', class_pred.shape)
        return class_pred
        

'''
class MultiChannelDiagnosticsModel(nn.Module):
    def __init__(self, num_classes, patch_size=16, embed_dim=128, num_heads=8, num_blocks=6):
        super(MultiChannelDiagnosticsModel, self).__init__()
        self.x_model = DiagnosticsModel(num_classes, patch_size, embed_dim, num_heads, num_blocks)
        self.y_model = DiagnosticsModel(num_classes, patch_size, embed_dim, num_heads, num_blocks)
    
    def forward(self, x, return_features=False):
        data_x = x[:, :, 0]  # x轴震荡数据
        data_y = x[:, :, 1]  # y轴震荡数据
        class_pred_x = self.x_model(data_x)
        class_pred_y = self.y_model(data_y)
        class_pred = (class_pred_x + class_pred_y) / 2
        
        if return_features:
            return class_pred, class_pred_x, class_pred_y
        else:
            return class_pred
'''