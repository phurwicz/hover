"""
Submodule that handles interaction with PyTorch.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from deprecated import deprecated


class VectorDataset(Dataset):
    """
    PyTorch Dataset of vectors.
    """

    DEFAULT_LOADER_KWARGS = dict(batch_size=64, shuffle=True, drop_last=False)

    def __init__(self, input_vectors, output_vectors):
        """Overrides the parent constructor."""
        assert len(input_vectors) == len(output_vectors)
        self.input_tensor = torch.FloatTensor(np.asarray(input_vectors))
        self.output_tensor = torch.FloatTensor(np.asarray(output_vectors))

    def __getitem__(self, index):
        """Makes the dataset an iterable."""
        return self.input_tensor[index], self.output_tensor[index], index

    def __len__(self):
        """Defines the length measure."""
        return len(self.input_tensor)

    def loader(self, **kwargs):
        keyword_args = self.__class__.DEFAULT_LOADER_KWARGS.copy()
        keyword_args.update(kwargs)
        return DataLoader(dataset=self, **keyword_args)


class MultiVectorDataset(Dataset):
    """
    PyTorch Dataset of vectors.
    """

    DEFAULT_LOADER_KWARGS = dict(batch_size=64, shuffle=True, drop_last=False)

    def __init__(self, input_vector_lists, output_vectors):
        """Overrides the parent constructor."""
        for _list in input_vector_lists:
            assert len(_list) == len(output_vectors)
        self.input_tensors = [
            torch.FloatTensor(np.asarray(_list)) for _list in input_vector_lists
        ]
        self.output_tensor = torch.FloatTensor(np.asarray(output_vectors))

    def __getitem__(self, index):
        """Makes the dataset an iterable."""
        input_vectors = [_tensor[index] for _tensor in self.input_tensors]
        return input_vectors, self.output_tensor[index], index

    def __len__(self):
        """Defines the length measure."""
        return len(self.output_tensor)

    def loader(self, **kwargs):
        keyword_args = self.__class__.DEFAULT_LOADER_KWARGS.copy()
        keyword_args.update(kwargs)
        return DataLoader(dataset=self, **keyword_args)


@deprecated(
    version="0.6.0",
    reason="will be removed in a future version; please use VectorDataset.loader() instead.",
)
def vector_dataloader(input_vectors, output_vectors, batch_size=64):
    """
    Loads data for training a torch nn.

    :param input_vectors: list of vectorized input.
    :type input_vectors: list of numpy.array
    :param output_vectors: list of vectorized output, e.g. classification labels in one-hot or probability vector format.
    :type output_vectors: list of numpy.array
    :param batch_size: size of each batch.
    :type batch_size: int
    """
    dataset = VectorDataset(input_vectors, output_vectors)
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )


def one_hot(encoded_labels, num_classes):
    """
    One-hot encoding into a float form.

    :param encoded_labels: integer-encoded labels.
    :type encoded_labels: list of int
    :param num_classes: the number of classes to encode.
    :type num_classes: int
    """
    return F.one_hot(torch.LongTensor(encoded_labels), num_classes=num_classes).float()


def label_smoothing(probabilistic_labels, coefficient=0.1):
    """
    Smooth probabilistic labels, auto-detecting the number of classes.

    :param probabilistic_labels: N by num_classes tensor
    :type probabilistic_labels: torch.Tensor or numpy.ndarray
    :param coefficient: the smoothing coeffient for soft labels.
    :type coefficient: float
    """
    assert (
        len(probabilistic_labels.shape) == 2
    ), f"Expected 2 dimensions, got shape {probabilistic_labels.shape}"
    assert coefficient >= 0.0, f"Expected non-negative smoothing, got {coefficient}"
    num_classes = probabilistic_labels.shape[-1]
    return (1.0 - coefficient) * probabilistic_labels + coefficient / num_classes
