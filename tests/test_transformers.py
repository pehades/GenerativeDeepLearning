from math import sin, cos

import pytest
import torch

from ml.transformers.models import AttentionBlock, MultiHeadAttention, PositionalEncoder


class TestTransformers:


    def test_attention_block(self):

        N = 100
        d_k = 64
        L = 256
        d_model = 256

        attention_block = AttentionBlock(d_k=d_k, d_model=d_model)

        X = torch.randn((N, L, d_model))

        output, attention = attention_block(X)

        assert output.shape == (N, L, d_k)

        # each attention column sums to 1. Applied the rounding due to floating errors (output is 0.99999 etc)
        assert (attention.detach().sum(axis=1).apply_(lambda x: round(x, 1)) == torch.ones((N, L))[0][0]).all()

    def test_multi_head_attention(self):

        N = 100
        d_k = 64
        L = 111
        n_heads = 8
        d_model = 512

        multi_head_attention = MultiHeadAttention(n_heads, d_k, d_model)

        X = torch.randn((N, L, d_model))
        output = multi_head_attention(X)

        assert output.shape == (N, L, d_model)

    def test_positional_encoding(self):
        L = 2
        d = 2
        x = torch.zeros(size=(2, L, d))

        positional_encoder = PositionalEncoder(L, d)
        output = positional_encoder(x)

        assert (output == torch.Tensor(
            [
                [
                    [sin(0), cos(0)],
                    [sin(1), cos(1/100)]
                ]
            ] * 2
        )).all()
