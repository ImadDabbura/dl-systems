import math
import array
from operator import mul
import numpy as np
from .autograd import Tensor
import struct
import gzip
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

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


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        ### BEGIN YOUR SOLUTION
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, 1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        if len(img.shape) < 3:
            size = int(math.sqrt(img.shape[0]))
            img = img.reshape(size, size, 1)
        h, w, _ = img.shape
        top_left = self.padding + shift_x
        bottom_left = self.padding + shift_y
        padded_img = np.pad(
            img,
            pad_width=(
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ),
        )
        return padded_img[
            top_left : top_left + h, bottom_left : bottom_left + w, :
        ]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        idxs = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        self.ordering = np.array_split(
            idxs,
            range(batch_size, len(dataset), batch_size),
        )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.pos = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.pos < len(self.ordering):
            out = self.dataset[self.ordering[self.pos]]
            self.pos += 1
            return tuple([Tensor(out[i]) for i in range(len(out))])
        else:
            raise StopIteration()
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.x, self.y = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.x[index]
        y = self.y[index]
        x = self.apply_transforms(x)
        return x, np.array(y)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.x)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
