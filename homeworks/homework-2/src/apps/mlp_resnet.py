import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Residual(fn)
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    total_loss = 0
    n = 0
    total_error = 0
    loss_func = nn.SoftmaxLoss()
    for i, (x, y) in enumerate(dataloader):
        m = y.shape[0]
        n += m
        preds = model(x)
        loss = loss_func(preds, y)
        total_loss += m * loss.detach().numpy()
        error = np.mean(preds.numpy().argmax(axis=1) != y.numpy())
        total_error += m * error
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
    return total_error / n, total_loss / n
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Datasets/Dataloaders
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz",
    )

    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size
    )
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    for _ in range(epochs):
        error, loss = epoch(train_dataloader, model, opt)
        train_loss.append(loss)
        train_accuracy.append(1 - error)
        error, loss = epoch(test_dataloader, model, None)
        test_loss.append(loss)
        test_accuracy.append(1 - error)
    return np.array(
        [train_accuracy[-1], train_loss[-1], test_accuracy[-1], test_loss[-1]]
    )
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
