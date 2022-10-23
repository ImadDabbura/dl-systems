import array
from operator import mul
import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


# From https://web.archive.org/web/20220509025752/http://yann.lecun.com/exdb/mnist/
DATA_TYPES = {
    0x08: "B",  # unsigned byte
    0x09: "b",  # signed byte
    0x0B: "h",  # short (2 bytes)
    0x0C: "i",  # int (4 bytes)
    0x0D: "f",  # float (4 bytes)
    0x0E: "d",
}  # double


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


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    m, k = y_one_hot.shape
    exp_z = ndl.exp(Z)
    log_softmax = Z - ndl.broadcast_to(
        ndl.reshape(ndl.log(ndl.summation(exp_z, (1,))), (m, 1)), (m, k)
    )
    return -(ndl.summation(log_softmax * y_one_hot)) / m
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    m = len(y)
    k = W2.shape[1]
    y_one_hot = np.zeros((m, k))
    y_one_hot[np.arange(m), y] = 1
    for i in range(0, m, batch):
        X_batch = X[i : i + batch]
        X_batch = ndl.Tensor(X_batch, requires_grad=False)
        y_batch = y_one_hot[i : i + batch]
        y_batch = ndl.Tensor(y_batch, requires_grad=False)
        bs = y_batch.shape[0]

        Z = ndl.relu(X_batch @ W1) @ W2
        loss = softmax_loss(Z, y_batch)
        loss.backward()
        W1_numpy = W1.numpy() - (lr * W1.grad.numpy())
        W2_numpy = W2.numpy() - (lr * W2.grad.numpy())
        W1 = ndl.Tensor(W1_numpy)
        W2 = ndl.Tensor(W2_numpy)
    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
