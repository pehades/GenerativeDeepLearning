import torch
from torch import nn

from ml.auto_encoders.auto_encoder import Decoder


class VariationalEncoder(nn.Module):
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
        self.fc_mean = nn.Linear(4 * 4 * 128, 2)
        self.fc_log_var = nn.Linear(4 * 4 * 128, 2)

    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor):

        batch = z_mean.shape[0]
        dim = z_mean.shape[1]

        # We send the epsilon to the same device as z_mean, z_log_var
        device = z_mean.device.type

        epsilon = torch.normal(mean=torch.zeros((batch, dim)), std=torch.ones((batch, dim))).to(device)

        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var, self.sample(z_mean, z_log_var)


class VariationalAutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = VariationalEncoder()
        self.decoder = Decoder()

    def forward(self, x):

        z_mean, z_log_var, z = self.encoder(x)
        return z_mean, z_log_var, self.decoder(z)
