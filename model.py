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
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # 类token（3维：适应批次输入）
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        # pos_embed = position_embedding(self.num_patches + 1, embed_dim)  # 计算正弦位置编码
        # self.register_buffer('pos_embed', pos_embed)  # 注册为buffer，不会被优化器更新

        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        x = x[:, :, 0]  # 选择第三列作为信号数据
        # x: (batch_size, window_size)
        batch_size = x.shape[0]
        patches = x.unfold(1, self.patch_size, self.patch_size)
        embeddings = self.proj(patches)  # (batch_size, num_patches, embed_dim)
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings += self.pos_embed
        return embeddings


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
        print(x.shape)
        print(x[:, 0, :])
        print(x[:, 1, :])
        attn_output, _ = self.self_attn(x, x, x)
        print(x[:, 0, :])
        print(x[:, 1, :])
        breakpoint()
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x

'''
class TransferableTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransferableTransformerBlock, self).__init__()
        self.msa = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.grl = GradientReversalLayer()
    
    def forward(self, x):
        # 多头自注意力
        attn_output, _ = self.msa(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_output
        
        # MLP
        mlp_output = self.mlp(self.ln2(x))
        x = x + mlp_output
        
        # 梯度反转层用于域适应
        reversed_x = self.grl(x)
        return x, reversed_x
    
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(DomainDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
    
    def forward(self, x):
        return x

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)
'''

class DiagnosticsModel(nn.Module):
    def __init__(self, num_classes, patch_size=16, embed_dim=128, num_heads=8, num_blocks=6):
        super(DiagnosticsModel, self).__init__()
        self.signal_embedding = SignalEmbedding(patch_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)])
        # self.transferable_transformer_blocks = TransferableTransformerBlock(embed_dim, num_heads)
        # self.domain_discriminator = DomainDiscriminator(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embed_dim, num_classes),
        )
    
    def forward(self, x, return_features=False):
        embeddings = self.signal_embedding(x)
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        # embeddings = self.transferable_transformer_blocks(embeddings)
        cls_tokens = embeddings[:, 0, :]
        # cls_tokens = embeddings.mean(dim = 1)
        # domain_pred = self.domain_discriminator(cls_tokens)
        class_pred = self.classifier(cls_tokens)
        if return_features:
            return class_pred, cls_tokens
        else:
            return class_pred