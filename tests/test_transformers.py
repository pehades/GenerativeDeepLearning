import pytest
import torch

from ml.transformers.models import AttentionBlock


class TestTransformers:


    def test_attention_block(self):

        N = 100
        d = 64
        L = 256
        input_dim = 32
        output_dim = 256

        attention_block = AttentionBlock(d=d, L=L, input_dim=input_dim, output_dim=128)

        X = torch.randn((N, L, input_dim))

        output, attention = attention_block(X)

        assert output.shape == (N, L, output_dim)

        # each attention column sums to 1. Applied the rounding due to floating errors (output is 0.99999 etc)
        assert (attention.detach().sum(axis=1).apply_(lambda x: round(x, 1)) == torch.ones((N, L))[0][0]).all()
