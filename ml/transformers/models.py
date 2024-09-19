import math

import torch
import torch.nn as nn


class AttentionBlock(nn.Module):

    def __init__(self, d: int, L: int, input_dim: int, output_dim: int):

        super().__init__()

        self.d = d
        self.W_Q = nn.Linear(input_dim, d)
        self.W_K = nn.Linear(input_dim, d)
        self.W_V = nn.Linear(input_dim, d)
        self.W_O = nn.Linear(d, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        # x has shape (N, L, input_dim)

        q = self.W_Q(x)  # shape of (N, L, d)
        k = self.W_K(x)  # shape of (N, L, d)
        v = self.W_V(x)  # shape of (N, L, d)

        attention = self.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d))  # shape of (N, L, L)

        values_with_attention = torch.bmm(attention, v)  # shape of (N, L, d)
        output = self.W_O(values_with_attention)  # shape of (N, L, output_dim)

        return output, attention



