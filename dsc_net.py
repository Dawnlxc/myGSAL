import torch
from torch import nn
import torch.nn.functional as F
import copy
import math
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

from config import Configure, ACT

'''
    This file define the architecture of the discriminator network (Fig.4)
'''

class LayerNorm(nn.Module):
    def __init__(self,
                 hidden_dims,
                 eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dims))
        self.beta = nn.Parameter(torch.zeros(hidden_dims))
        self.eps = eps

    def forward(self, x):
        # BN -> Normalize by feature (along the Batch direction)
        # LN -> Normalize for  (along the feature direction)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, unbiased=False, keepdim=True)
        res = self.gamma * (x - mean) / (torch.sqrt(std + self.eps)) + self.beta
        return res


class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_emb = nn.Embedding(config.max_pos_emb, config.hidden_dims)
        self.layer_norm = LayerNorm(hidden_dims=config.hidden_dims,
                                    eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_drop_rate)

    def forward(self, input):
        shape = input.shape
        seg_len = shape[1]
        device = input.device

        pos_idx = torch.arange(seg_len, dtype=torch.long, device=device)
        pos_idx = pos_idx.unsqueeze(0).expand(shape[:2])

        pos_emb_out = self.pos_emb(pos_idx)

        emb = input + pos_emb_out
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class MSA(nn.Module):
    """
      Multi-head self-attention block
    """

    def __init__(self, config):
        super(MSA, self).__init__()

        self.n_attention_heads = config.n_attention_heads
        self.attention_head_size = int(config.hidden_dims / config.n_attention_heads)
        self.all_heads_size = self.n_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_dims, self.all_heads_size)
        self.key = nn.Linear(config.hidden_dims, self.all_heads_size)
        self.value = nn.Linear(config.hidden_dims, self.all_heads_size)

        self.dropout = nn.Dropout(config.attention_drop_rate)

    def reshape(self, x):
        shape = x.size()[:-1] + (self.n_attention_heads, self.attention_head_size)
        x = x.view(*shape)
        # x -> (B, n_attention_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x_query = self.reshape(self.query(x))
        x_key = self.reshape(self.key(x))
        x_value = self.reshape(self.value(x))

        att_score = torch.matmul(x_query, x_key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        att_prob = self.dropout(nn.Softmax(dim=-1)(att_score))

        attention = torch.matmul(att_prob, x_value)
        attention = attention.permute(0, 2, 1, 3).contiguous()

        target_shape = attention.size()[:-2] + (self.all_heads_size,)
        attention = attention.view(*target_shape)

        return attention


class MSABlock(nn.Module):
    """
        MSA Block = MSA + Linear projection + LayerNorm
    """
    def __init__(self, config):
        super(MSABlock, self).__init__()
        self.block = nn.Sequential(
            MSA(config),
            nn.Linear(config.hidden_dims, config.hidden_dims),
            nn.Dropout(config.hidden_drop_rate),
        )
        self.layernorm = LayerNorm(config.hidden_dims, eps=config.layer_norm_eps)

    def forward(self, x):
        x_tmp = self.block(x)
        output = self.layernorm(x_tmp + x)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dims, config.intermediate_dims),
            nn.GELU(),
            nn.Linear(config.intermediate_dims, config.hidden_dims),
            nn.Dropout(config.hidden_drop_rate)
        )
        self.layernorm = LayerNorm(config.hidden_dims, eps=config.layer_norm_eps)

    def forward(self, x):
        x_tmp = self.mlp(x)
        output = self.layernorm(x_tmp + x)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.msa_block = MSABlock(config)
        self.mlp = MLP(config)

    def forward(self, x):
        return self.mlp(self.msa_block(x))


class TransformerLayersBlock(nn.Module):
    '''
        Transformer Layer Block:
            TransformerLayer * L
        Args:
            Configure
        Return:
            A numpy list that saved selected output for each layer
    '''

    def __init__(self, config, is_seg=True):
        super(TransformerLayersBlock, self).__init__()
        self.is_seg = is_seg
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.n_hidden_layers)]
        )

    def forward(self, x):
        outs = list()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.is_seg:
                outs.append(x)
        if not self.is_seg:
            outs.append(x)
        return outs

class TransformerModule(nn.Module):
    '''
        Transformer Module = Linear projection + LayerNorm + Position Embedding + TransformerLayersBlock
        Return: outs -> numpy list
                if self.is_seg == True:
                    return [out_1, out_2, ..., out_L] or else [out_L]
    '''
    def __init__(self, config, is_seg=True):
        super(TransformerModule, self).__init__()
        self.is_seg = is_seg
        self.linear_proj = nn.Sequential(nn.Linear(config.patch_size[0] * config.patch_size[1] * config.in_channels, config.hidden_dims),
                                         nn.GELU())
        self.layernorm = LayerNorm(config.hidden_dims, eps=config.layer_norm_eps)
        self.embedding = PositionEmbedding(config)
        self.transformer_layers = TransformerLayersBlock(config, is_seg=is_seg)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.layernorm(x)
        x = self.embedding(x)
        outs = self.transformer_layers(x)
        # if not self.is_seg:
        #     return outs[-1]
        return outs

class TransformerEncoder(nn.Module):
    '''
        Transformer Encoder = Patch Embedding + Transformer Module + Feature Projection
        Return: last_trans_out -> (B, H//ph * W//pw, ph * pw * C=1)
                x -> (B, hidden_dims, H/2**4, H/2**4)
    '''
    def __init__(self, config, is_seg=True):
        super().__init__()
        self.transformer_module = TransformerModule(config, is_seg)
        self.sample_rate = config.sample_rate
        self.is_seg = is_seg
        self.sample_v = int(math.pow(2, self.sample_rate))
        self.feature_proj = nn.Linear(config.hidden_dims,
                                      config.patch_size[0] *
                                      config.patch_size[1] *
                                      config.hidden_dims // (self.sample_v**2))
        self.hh = config.patch_size[0] // self.sample_v
        self.ww = config.patch_size[1] // self.sample_v

        self.patch_size = config.patch_size

    def forward(self, x):
        '''
            Args: x -> (B, C, H, W)
        '''
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        x = x.view(B, C, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, -1, C * ph * pw)
        # Last output of transformer layers block
        last_trans_out = self.transformer_module(x)[-1]

        if not self.is_seg:
            return last_trans_out
        x = self.feature_proj(last_trans_out)
        # x -> (B, H//ph * W//pw, config.hidden_dims * ph * pw)
        x = x.view(B, H//ph, W//pw, ph, pw, -1)
        x = x.permute(0, 5, 1, 3, 2, 4)
        # x = x.reshape(B, -1, H//ph * ph, W//pw * pw)
        x = x.reshape(B, -1, H//ph * self.hh, W//pw * self.ww)
        return last_trans_out, x

class VitDiscriminator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_size,
                 hidden_dims=1024,
                 n_hidden_layers=8,
                 n_attention_heads=16,
                 sample_rate=4):
        super(VitDiscriminator, self).__init__()

        # TODO: Define the configure here
        config = Configure(in_channels=in_channels,
                           out_channels=out_channels,
                           patch_size=patch_size,
                           hidden_dims=hidden_dims,
                           n_hidden_layers=n_hidden_layers,
                           n_attention_heads=n_attention_heads,
                           sample_rate=sample_rate)
        self.encoder = TransformerEncoder(config, is_seg=True)
        self.classifier = nn.Linear(hidden_dims, out_channels)

    def forward(self, x):
        x = self.encoder(x)
        '''
            x -> tuple; ((B, -1, patch_size[0]*patch_size[1]*C), (B, 1, H, W))
            if self.is_seg == True:
                x[1] is used to calculate the L1 loss
            pass x[0] to the classifier
        '''
        print(x[0].shape, x[1].shape)
        x, aux_x = x[0].mean(dim=1), x[1]
        x = self.classifier(x)

        return x

class Discriminator(nn.Module):
    '''
        Returns:
            output -> (B, -1) used to compute L1 loss
            output1 -> (B, 1) classification result
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_size,
                 hidden_dims=1024,
                 n_hidden_layers=8,
                 n_attention_heads=16,
                 # decoder_features=[512, 256, 128, 64],
                 sample_rate=4):
        super(Discriminator, self).__init__()
        config = Configure(in_channels=in_channels,
                           out_channels=out_channels,
                           patch_size=patch_size,
                           hidden_dims=hidden_dims,
                           n_hidden_layers=n_hidden_layers,
                           n_attention_heads=n_attention_heads,
                           sample_rate=sample_rate)
        self.encoder = TransformerEncoder(config, is_seg=True)
        self.cls = nn.Sequential(
            nn.Linear(1024*16*16, 100),
            # nn.BatchNorm1d(100),
            nn.GELU(),
            nn.Linear(100, 2),
            # nn.BatchNorm1d(2),
            nn.GELU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def get_n_flat_features(self, x):
        size = x.size()[1:]
        n = 1
        for s in size:
            n *= s
        return n
    def forward(self, x):
        B, C, H, W = x.shape
        _, x = self.encoder(x)
        output = x.view(B, -1)
        output1 = output.view(-1, self.get_n_flat_features(output))
        output1 = self.cls(output1)
        return output, output1


if __name__ == '__main__':

    test = torch.ones(10, 1, 256, 256)
    model = Discriminator(1, 1, (32, 32))
    out = model(test)