
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from ml.transformers.models import Transformer


if __name__ == '__main__':
    """
    training with mock data to confirm that error is reducing
    """
    epochs = 10_000

    N = 10
    L = 10
    d = 32
    n_heads = 2

    source_vocab_size = 50
    target_vocab_size = 50

    X = torch.randint(low=0, high=source_vocab_size, size=(N, L)).to(torch.long)
    Y = torch.randint(low=0, high=target_vocab_size, size=(N, L)).to(torch.long)

    transformer = Transformer(source_vocab_size, target_vocab_size, L, d, n_heads)
    loss_function = CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(transformer.parameters(), lr=0.0001)

    start_token = target_vocab_size - 2
    end_token = target_vocab_size - 1


    for epoch in range(epochs):

        Y = torch.cat([Y[:, :-1], end_token * torch.ones(size=(N, 1))], dim=1).to(torch.long)

        target = torch.cat([start_token * torch.ones(size=(N, 1)), Y[:, 1:]], dim=1).to(torch.long)

        output = transformer(X, Y)

        optimizer.zero_grad()
        loss = loss_function(output.contiguous().view(-1, target_vocab_size), target.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(float(loss))



