import torch.nn as nn

import torch


class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(0.3)

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.batch_2 = nn.BatchNorm2d(128, momentum=0.9)
        self.leaky_relu_2 = nn.LeakyReLU(0.2)
        self.dropout_2 = nn.Dropout(0.3)

        self.conv_3 = nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1)
        self.batch_3 = nn.BatchNorm2d(512, momentum=0.9)
        self.leaky_relu_3 = nn.LeakyReLU(0.2)
        self.dropout_3 = nn.Dropout(0.3)

        self.conv_4 = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.dropout_1(self.leaky_relu_1(self.conv_1(x)))
        x = self.dropout_2(self.leaky_relu_2(self.batch_2(self.conv_2(x))))
        x = self.dropout_3(self.leaky_relu_3(self.batch_3(self.conv_3(x))))

        x = self.fc(self.flatten(self.conv_4(x)))
        return self.sigmoid(x)


class Generator(nn.Module):

    def __init__(self, latent_dim: int = 100):

        super().__init__()

        self.conv_1_transpose = nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0)
        self.batch_1 = nn.BatchNorm2d(512, momentum=0.9)
        self.leaky_relu_1 = nn.LeakyReLU(0.2)

        self.conv_2_transpose = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.batch_2 = nn.BatchNorm2d(256, momentum=0.9)
        self.leaky_relu_2 = nn.LeakyReLU(0.2)

        self.conv_3_transpose = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.batch_3 = nn.BatchNorm2d(128, momentum=0.9)
        self.leaky_relu_3 = nn.LeakyReLU(0.2)

        self.conv_4_transpose = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.batch_4 = nn.BatchNorm2d(64, momentum=0.9)
        self.leaky_relu_4 = nn.LeakyReLU(0.2)

        self.conv_5_transpose = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):

        x = x.reshape((-1, 100, 1, 1))

        x = self.leaky_relu_1(self.batch_1(self.conv_1_transpose(x)))
        x = self.leaky_relu_2(self.batch_2(self.conv_2_transpose(x)))
        x = self.leaky_relu_3(self.batch_3(self.conv_3_transpose(x)))
        x = self.leaky_relu_4(self.batch_4(self.conv_4_transpose(x)))

        return self.conv_5_transpose(x)
