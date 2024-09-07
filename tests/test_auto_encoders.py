import torch

from ml.auto_encoders.auto_encoder import Encoder, Decoder, AutoEncoder
from ml.auto_encoders.variational_auto_encoder import VariationalEncoder, VariationalAutoEncoder


class TestAutoEncoders:

    def test_encoder(self):

        encoder = Encoder()

        x = torch.ones((64, 1, 32, 32))

        y = encoder(x)

        assert y.shape == (64, 2)

    def test_decoder(self):

        decoder = Decoder()

        x = torch.ones((64, 2))
        y = decoder(x)

        assert y.shape == (64, 1, 32, 32)

    def test_auto_encoder(self):

        auto_encoder = AutoEncoder()

        x = torch.ones((32, 1, 32, 32))

        assert auto_encoder(x).shape == x.shape

    def test_variational_encoder(self):

        variational_encoder = VariationalEncoder()

        x = torch.ones((32, 1, 32, 32))

        _, _, output = variational_encoder(x)
        assert output.shape == (32, 2)

    def test_variational_auto_encoder(self):

        variational_auto_encoder = VariationalAutoEncoder()

        x = torch.ones((32, 1, 32, 32))
        _, _, output = variational_auto_encoder(x)
        assert output.shape == (32, 1, 32, 32)