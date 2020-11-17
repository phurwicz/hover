"""
Submodule that handles interaction with PyTorch.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class VectorDataset(Dataset):
    """
    PyTorch Dataset of vectors.
    """

    def __init__(self, input_vectors, output_vectors):
        assert len(input_vectors) == len(input_vectors)
        self.input_tensor = torch.FloatTensor(input_vectors)
        self.output_tensor = torch.FloatTensor(output_vectors)

    def __getitem__(self, index):
        return self.input_tensor[index], self.output_tensor[index], index

    def __len__(self):
        return len(self.input_tensor)


def vector_dataloader(input_vectors, output_vectors, batch_size=64):
    """
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
    :param encoded_labels: integer-encoded labels.
    :type encoded_labels: list of int
    :param num_classes: the number of classes to encode.
    :type num_classes: int
    """
    return F.one_hot(torch.LongTensor(encoded_labels), num_classes=num_classes).float()


def label_smoothing(probabilistic_labels, num_classes, coefficient=0.1):
    """
    :param probabilistic_labels: N by num_classes tensor
    :type probabilistic_labels: torch.Tensor or numpy.ndarray
    :param num_classes: the number of classes to encode.
    :type num_classes: int
    :param coefficient: the smoothing coeffient for soft labels.
    :type coefficient: float
    """
    assert (
        len(probabilistic_labels.shape) == 2
    ), f"Expected 2 dimensions, got shape {probabilistic_labels.shape}"
    assert (
        probabilistic_labels.shape[-1] == num_classes
    ), f"Expected {num_classes} classes, got {probabilistic_labels.shape[-1]}"
    return (1.0 - coefficient) * probabilistic_labels + coefficient / num_classes
