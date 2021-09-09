"""
Submodule that handles interaction with PyTorch.
"""
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
        self.input_tensor = torch.FloatTensor(input_vectors)
        self.output_tensor = torch.FloatTensor(output_vectors)

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
        self.input_tensors = [torch.FloatTensor(_list) for _list in input_vector_lists]
        self.output_tensor = torch.FloatTensor(output_vectors)

    def __getitem__(self, index):
        """Makes the dataset an iterable."""
        input_vectors = [_tensor[index] for _tensor in self.input_tensors]
        return input_vectors, self.output_tensor[index], index

    def __len__(self):
        """Defines the length measure."""
        return len(self.input_tensor)

    def loader(self, **kwargs):
        keyword_args = self.__class__.DEFAULT_LOADER_KWARGS.copy()
        keyword_args.update(kwargs)
        return DataLoader(dataset=self, **keyword_args)


@deprecated(
    version="0.5.1",
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


def cross_entropy_with_probs(logits, target, weight=None, reduction="mean"):
    """
    Cherry-picked from snorkel.classification.
    Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    """
    num_points, num_classes = logits.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = logits.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = logits.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(logits, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


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
