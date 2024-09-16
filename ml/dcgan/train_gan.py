import os
import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Compose, CenterCrop, Normalize

from ml import ROOT_DIR
from ml.dcgan.models import Generator, Discriminator


def load_celeb_a(batch_size):

    dataset = ImageFolder(root=os.path.join(ROOT_DIR, 'data', 'img_align_celeba'),
                          transform=Compose([
                              Resize(64),
                              CenterCrop(64),
                              ToTensor(),
                              Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))

    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


def plot_images(model: torch.nn.Module, fixed_noise: torch.Tensor):

    number_of_images = len(fixed_noise)

    with torch.no_grad():
        predicted_images = model(fixed_noise).to('cpu')
        predicted_images = (predicted_images + 1) / 2

    fig, ax = plt.subplots(number_of_images, 1)
    for index in range(number_of_images):
        predicted_image = predicted_images[index]
        ax[index].imshow(np.asarray(predicted_image).transpose(1, 2, 0))

    plt.show()


def main():
    epochs = 8
    batch_size = 128
    lr = 0.0002
    beta1 = 0.5
    betas = (beta1, 0.999)

    latent_dim = 100
    n_channels = 3

    dataloader = load_celeb_a(batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator = Generator(latent_dim=latent_dim, n_channels=n_channels).to(device)
    discriminator = Discriminator(n_channels=n_channels).to(device)


    generator_optim = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    loss_fn = nn.BCELoss()

    fixed_noise = torch.normal(mean=0, std=1, size=(4, 100))

    for epoch in range(epochs):
        start = time.time()
        for i, (X, _) in enumerate(dataloader):
            new_start = time.time()

            X = X.to(device)

            # discriminator train
            discriminator_optim.zero_grad()

            real_images = X
            real_output = discriminator(real_images).view(-1)

            label = torch.ones(size=(len(X), )).to(device)
            real_loss = loss_fn(real_output, label)
            real_loss.backward()

            random_vectors = torch.randn(len(X), 100, 1, 1, device=device)
            fake_images = generator(random_vectors)
            fake_output = discriminator(fake_images.detach()).view(-1)

            label = torch.zeros(size=(len(X), )).to(device)
            fake_loss = loss_fn(fake_output, label)
            fake_loss.backward()

            d_loss = (fake_loss + real_loss) / 2
            discriminator_optim.step()

            # generator train
            generator_optim.zero_grad()

            label = torch.ones(size=(len(X),)).to(device)
            predictions = discriminator(fake_images).view(-1)


            generator_loss = loss_fn(predictions, label)
            generator_loss.backward()
            generator_optim.step()


            print('discriminator', d_loss.real, 'generator', generator_loss, 'time', time.time() - new_start)
            if i % 30 == 0:
                plot_images(generator, fixed_noise=fixed_noise.to(device))


        print("time total", time.time() - start)


if __name__ == '__main__':

    main()



