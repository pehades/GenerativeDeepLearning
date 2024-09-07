import torch
from torch import nn


class Encoder(nn.Module):
    """
    Assumes 32x32 pictures
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(4 * 4 * 128, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 4 * 4 * 128)

        self.conv_transpose_1 = nn.ConvTranspose2d(128, 128, (3, 3), stride=2, padding=1, output_padding=1)
        self.conv_transpose_2 = nn.ConvTranspose2d(128, 64, (3, 3), stride=2, padding=1, output_padding=1)
        self.conv_transpose_3 = nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(32, 1, kernel_size=(1, 1), padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.fc(x)
        x = torch.reshape(x, (-1, 128, 4, 4))
        x = self.relu(self.conv_transpose_1(x))
        x = self.relu(self.conv_transpose_2(x))
        x = self.relu(self.conv_transpose_3(x))
        x = self.sigmoid(self.conv(x))
        return x


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


