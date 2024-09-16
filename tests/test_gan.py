import pytest
import torch

from ml.dcgan.models import Discriminator, Generator


class TestGAN:

    def test_discriminator(self):

        x = torch.ones((16, 1, 64, 64))

        discriminator = Discriminator(n_channels=1)

        output = discriminator(x)

        assert output.shape == (16, 1, 1, 1)

    def test_generator(self):

        x = torch.ones((16, 100))

        generator = Generator(latent_dim=100, n_channels=1)
        output = generator(x)

        assert output.shape == (16, 1, 64, 64)
