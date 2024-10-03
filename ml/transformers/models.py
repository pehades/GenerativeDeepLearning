import math

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

    def forward(self, x, mask: bool = True):

        # x has shape (N, L, d_model)

        q = self.W_Q(x)  # shape of (N, L, d_k)
        k = self.W_K(x)  # shape of (N, L, d_k)
        v = self.W_V(x)  # shape of (N, L, d_k)

        L = x.shape[1]
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

    def forward(self, x, mask: bool = True):

        # x has shape (N, L, d_model)
        attention_heads = torch.cat(
            [attention_head(x, mask)[0] for attention_head in self.attention_heads], dim=-1
        )
        output = self.W_concat(attention_heads)
        return output

