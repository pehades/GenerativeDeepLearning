import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Pad


def get_dataloader(batch_size: int):

    training_data = FashionMNIST(
        '../data',
        download=True,
        train=True,
        transform=Compose(
            [ToTensor(), Pad(2)]
        )
    )

    return DataLoader(training_data, batch_size=batch_size), training_data


def plot_images(model: torch.nn.Module, X: torch.Tensor, device: str):

    number_of_images = X.shape[0]

    model.eval()
    with torch.no_grad():
        _, _, predicted_images = model(
            X.reshape((number_of_images, 1, 32, 32)).to(device)
        )

        predicted_images = predicted_images.reshape(number_of_images, 32, 32).to('cpu')

    model.train()

    fig, ax = plt.subplots(number_of_images, 2)

    for index in range(number_of_images):

        ax[index, 0].imshow(np.asarray(X[index]), cmap='gray', vmin=0, vmax=1)
        ax[index, 1].imshow(np.asarray(predicted_images[index]), cmap='gray', vmin=0, vmax=1)

    plt.show()
