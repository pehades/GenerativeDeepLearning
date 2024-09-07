import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.transforms import Pad, Compose, ToTensor

from ml import ROOT_DIR
from ml.auto_encoders.auto_encoder import AutoEncoder
from ml.auto_encoders.variational_auto_encoder import VariationalAutoEncoder


def main():

    training_data = get_training_data()

    class_indices, total_sampled_indices = get_random_sample_indices(training_data, elements_per_class=50)

    X = torch.cat([training_data[i][0] for i in total_sampled_indices], axis=0)

    auto_encoder = load_auto_encoder()

    with torch.no_grad():
        z_mean, z_log_var, z = auto_encoder.encoder(
            X.reshape((-1, 1, 32, 32))
        )

    plt.scatter(z[:, 0], z[:, 1], c=class_indices)
    plt.show()


def load_auto_encoder():
    model_path = 'auto_encoder.pt'
    auto_encoder = VariationalAutoEncoder()
    auto_encoder.load_state_dict(torch.load(model_path))
    auto_encoder.eval()
    return auto_encoder


def get_random_sample_indices(training_data: MNIST, elements_per_class: int):
    labels = training_data.train_labels

    class_indices = []
    total_sampled_indices = []
    for i in range(len(torch.unique(labels))):
        indices = torch.where(labels == i)[0]
        sampled_indices = np.random.choice(indices, elements_per_class, replace=False)
        class_indices.extend([i] * elements_per_class)
        total_sampled_indices.extend(sampled_indices)

    return class_indices, total_sampled_indices


def get_training_data() -> FashionMNIST:
    training_data = FashionMNIST(
        root=os.path.join(ROOT_DIR, 'data'),
        train=True,
        transform=Compose(
            [ToTensor(), Pad(2)]
        )
    )
    return training_data


if __name__ == '__main__':
    main()
