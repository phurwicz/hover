"""
Snorkel is omitted from hover's dependencies due to its overly strict requirement specification, which can break Anaconda builds when it doesn't have to.
"""
import torch
import torch.nn.functional as F


def cross_entropy_with_probs(logits, target, weight=None, reduction="mean"):
    """
    Cherry-picked from snorkel.classification.
    Calculate cross-entropy loss when targets are probabilities (floats), not ints.

    Original: https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.cross_entropy_with_probs.html#snorkel.classification.cross_entropy_with_probs
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
