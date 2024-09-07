import pytest
import torch

from ml.gan.models import Discriminator, Generator


class TestGAN:

    def test_discriminator(self):

        x = torch.ones((16, 1, 64, 64))

        discriminator = Discriminator()

        output = discriminator(x)

        assert output.shape == (16, 1)

    def test_generator(self):

        x = torch.ones((16, 100))

        generator = Generator()
        output = generator(x)

        assert output.shape == (16, 1, 64, 64)