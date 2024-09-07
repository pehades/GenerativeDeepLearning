import torch.cuda
from torch.nn import MSELoss
from torch.optim import Adam

from ml.auto_encoders.auto_encoder import AutoEncoder
from ml.auto_encoders.training_utils import get_dataloader, plot_images
from ml.auto_encoders.variational_auto_encoder import VariationalAutoEncoder


def main(model_path: str, train_loop, model):

    # Bigger batch size is better
    epochs = 10
    batch_size = 1024


    dataloader, training_data = get_dataloader(batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.01)


    indices = [0, 1, 2]
    tensors_to_plot = None #torch.cat([training_data[i][0] for i in indices], axis=0)

    train_loop(epochs, model, optimizer, dataloader, device, tensors_to_plot)

    torch.save(model.state_dict(), model_path)


def auto_encoder_train_loop(epochs, model, optimizer, dataloader, device, tensors_to_plot):
    loss_fn = MSELoss()
    for epoch in range(epochs):

        if tensors_to_plot:
            plot_images(model, tensors_to_plot, device)

        running_loss = 0
        for X, _ in dataloader:
            optimizer.zero_grad()

            X = X.to(device)

            output = model(X)
            loss = loss_fn(
                output.flatten(1, -1), X.flatten(1, -1)
            )
            loss.backward()
            optimizer.step()

            running_loss += loss
        print(running_loss)


def variational_auto_encoder_train_loop(epochs, model, optimizer, dataloader, device, tensors_to_plot: torch.Tensor | None):

    reconstruction_loss = MSELoss()

    for epoch in range(epochs):

        if tensors_to_plot is not None:
            plot_images(model, tensors_to_plot, device)

        running_loss = 0
        for X, _ in dataloader:
            optimizer.zero_grad()

            X = X.to(device)

            z_mean, z_log_var, output = model(X)

            loss_1 = reconstruction_loss(
                output.flatten(1, -1), X.flatten(1, -1)
            )

            loss_2 = torch.mean(
                torch.sum(
                    -0.5 * (1 + z_log_var - (z_mean ** 2) - torch.exp(z_log_var)),
                    dim=1
                )
            )
            loss = loss_1 + torch.clip(loss_2, max=loss_1)

            loss.backward()
            optimizer.step()

            running_loss += loss
        print(running_loss)


if __name__ == '__main__':
    model_path = 'auto_encoder.pt'
    main(model_path, train_loop=variational_auto_encoder_train_loop, model=VariationalAutoEncoder())
