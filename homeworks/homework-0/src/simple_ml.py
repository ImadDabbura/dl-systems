import array
from operator import mul
import struct
import numpy as np
import gzip

try:
    from simple_ml_ext import *
except:
    pass


# From https://web.archive.org/web/20220509025752/http://yann.lecun.com/exdb/mnist/
DATA_TYPES = {
    0x08: "B",  # unsigned byte
    0x09: "b",  # signed byte
    0x0B: "h",  # short (2 bytes)
    0x0C: "i",  # int (4 bytes)
    0x0D: "f",  # float (4 bytes)
    0x0E: "d",
}  # double


def add(x, y):
    """A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_images_file(image_file):
    fd = gzip.open(image_file, "rb")
    zeros, data_type, n_dimensions = struct.unpack(">HBB", fd.read(4))
    dimension_sizes = struct.unpack(
        f"> {'I' * n_dimensions}", fd.read(4 * n_dimensions)
    )
    data = array.array(DATA_TYPES[data_type], fd.read())
    return (
        np.array(data, dtype=np.float32).reshape(-1, mul(*dimension_sizes[1:]))
        / 255
    )


def parse_labels_file(label_filename):
    fd = gzip.open(label_filename, "rb")
    zeros, data_type, n_dimensions = struct.unpack(">HBB", fd.read(4))
    fd.seek(4 * n_dimensions, 1)
    labels = array.array(DATA_TYPES[data_type], fd.read())
    return np.array(labels, np.uint8)


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    X = parse_images_file(image_filename)
    y = parse_labels_file(label_filename)
    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    log_softmax = np.log(np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True))
    return -np.mean(log_softmax[range(len(y)), y])
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    m = len(y)
    k = np.max(y) + 1
    for i in range(0, m, batch):
        # Forward
        X_batch = X[i : i + batch]
        y_batch = y[i : i + batch]
        h = X_batch.dot(theta)
        Z = np.exp(h) / np.sum(np.exp(h), axis=1, keepdims=True)

        # Backward
        bs = len(y_batch)
        I = np.zeros((bs, k))
        I[np.arange(bs), y_batch] = 1
        grad = X_batch.transpose().dot(Z - I)
        theta -= (lr * grad) / bs
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    m = len(y)
    k = np.max(y) + 1
    ### BEGIN YOUR CODE
    for i in range(0, m, batch):
        X_batch = X[i : i + batch]
        y_batch = y[i : i + batch]
        bs = len(y_batch)

        # Forward
        # Layer 1
        h1 = X_batch.dot(W1)
        Z1 = np.maximum(0, h1)
        assert h1.shape == Z1.shape

        # Layer 2
        h2 = Z1.dot(W2)
        Z2 = np.exp(h2) / np.sum(np.exp(h2), axis=1, keepdims=True)
        assert h2.shape == Z2.shape

        # Backward
        # Layer 2
        I = np.zeros((bs, k))
        assert Z2.shape == I.shape
        I[np.arange(bs), y_batch] = 1

        W2_grad = Z1.transpose().dot(Z2 - I)
        assert W2.shape == W2_grad.shape
        Z1_grad = (Z2 - I).dot(W2.transpose())
        W2 -= (lr * W2_grad) / bs

        # Layer 1
        W1_grad = X_batch.transpose().dot(Z1_grad * (Z1 > 0))
        assert W1.shape == W1_grad.shape
        W1 -= (lr * W1_grad) / bs
    ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(
    X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100, cpp=False
):
    """Example function to fully train a softmax regression classifier"""
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print(
            "|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |".format(
                epoch, train_loss, train_err, test_loss, test_err
            )
        )


def train_nn(
    X_tr, y_tr, X_te, y_te, hidden_dim=500, epochs=10, lr=0.5, batch=100
):
    """Example function to train two layer neural network"""
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(
        hidden_dim
    )
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print(
            "|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |".format(
                epoch, train_loss, train_err, test_loss, test_err
            )
        )


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    X_te, y_te = parse_mnist(
        "data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz"
    )

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
