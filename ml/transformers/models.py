import math
from math import sin, cos
from typing import Optional

import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """
    Takes an input with d_model dimension. Creates Q, K, V and produces an output with d_k dimension
    """
    def __init__(self, d_k: int, d_model: int):

        super().__init__()

        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, source, target: Optional[torch.Tensor] = None, mask: bool = True):

        if target is None:
            target = source

        # x has shape (N, L, d_model)

        q = self.W_Q(source)  # shape of (N, L, d_k)
        k = self.W_K(source)  # shape of (N, L, d_k)
        v = self.W_V(target)  # shape of (N, L, d_k)

        L = source.shape[1]
        zeros = torch.zeros((L, L))
        if mask:
            mask_tensor = torch.triu(zeros - torch.inf, diagonal=1)
        else:
            mask_tensor = zeros

        # shape of (N, L, L)
        attention = self.softmax(mask_tensor + torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k))

        output = torch.bmm(attention, v)  # shape of (N, L, d_k)

        return output, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads: int, d_k: int, d_model: int):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [AttentionBlock(d_k, d_model) for _ in range(n_heads)]
        )
        self.W_concat = nn.Linear(d_k * n_heads, d_model)

    def forward(self, source, target: Optional[torch.Tensor] = None, mask: bool = True):

        if target is None:
            target = source

        # x has shape (N, L, d_model)
        attention_heads = torch.cat(
            [attention_head(source, target, mask=mask)[0] for attention_head in self.attention_heads], dim=-1
        )
        output = self.W_concat(attention_heads)
        return output


class PositionalEncoder(nn.Module):

    def __init__(self, L: int, d: int, n: int = 10_000):
        super().__init__()
        # k / n ^(2*i / d) k mexri L, i mexri d/2
        self.P = torch.zeros(size=(L, d))
        for k in range(L):
            for i in range(int(d/2)):
                self.P[k, 2*i] = sin(k / (n ** (2 * i / d)))
                self.P[k, 2*i + 1] = cos(k / (n ** (2 * i / d)))

    def forward(self, x):
        return x + self.P


class Encoder(nn.Module):

    def __init__(self, dictionary_dim: int, L: int, d: int, n_heads: int):
        super().__init__()
        self.embedding = nn.Embedding(dictionary_dim, d)
        self.multi_head_attention = MultiHeadAttention(n_heads=n_heads, d_k=d, d_model=d)
        self.positional_encoder = PositionalEncoder(L, d)

    def forward(self, x):

        # x has dimension (N, L, dictionary_dim)
        x = self.embedding(x.to(torch.int64))
        x = self.multi_head_attention(x)
        return self.positional_encoder(x)
