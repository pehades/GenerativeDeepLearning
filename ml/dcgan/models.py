import torch.nn as nn

import torch


class Discriminator(nn.Module):

    def __init__(self, n_channels: int = 3):

        super().__init__()

        self.conv_1 = nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.leaky_relu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_2 = nn.BatchNorm2d(128)
        self.leaky_relu_2 = nn.LeakyReLU(0.2)

        self.conv_3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_3 = nn.BatchNorm2d(256)
        self.leaky_relu_3 = nn.LeakyReLU(0.2)

        self.conv_4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_4 = nn.BatchNorm2d(512)
        self.leaky_relu_4 = nn.LeakyReLU(0.2)

        self.conv_5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu_1(self.conv_1(x))
        x = self.leaky_relu_2(self.batch_2(self.conv_2(x)))
        x = self.leaky_relu_3(self.batch_3(self.conv_3(x)))
        x = self.leaky_relu_4(self.batch_4(self.conv_4(x)))
        x = self.sigmoid(self.conv_5(x))
        return x


class Generator(nn.Module):

    def __init__(self, latent_dim: int = 100, n_channels: int = 3):

        super().__init__()

        self.conv_1_transpose = nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(512)
        self.leaky_relu_1 = nn.LeakyReLU(0)

        self.conv_2_transpose = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_2 = nn.BatchNorm2d(256)
        self.leaky_relu_2 = nn.LeakyReLU(0)

        self.conv_3_transpose = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_3 = nn.BatchNorm2d(128)
        self.leaky_relu_3 = nn.LeakyReLU(0)

        self.conv_4_transpose = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_4 = nn.BatchNorm2d(64)
        self.leaky_relu_4 = nn.LeakyReLU(0)

        self.conv_5_transpose = nn.ConvTranspose2d(64, n_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = x.reshape((-1, 100, 1, 1))

        x = self.leaky_relu_1(self.batch_1(self.conv_1_transpose(x)))
        x = self.leaky_relu_2(self.batch_2(self.conv_2_transpose(x)))
        x = self.leaky_relu_3(self.batch_3(self.conv_3_transpose(x)))
        x = self.leaky_relu_4(self.batch_4(self.conv_4_transpose(x)))

        return self.tanh(self.conv_5_transpose(x))
