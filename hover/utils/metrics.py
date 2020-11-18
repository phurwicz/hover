import numpy as np


def classification_accuracy(true, pred):
    """
    Accuracy measure on two arrays. Intended for classification problems.
    :param true: true labels.
    :type true: Numpy array
    :param pred: predicted labels.
    :type pred: Numpy array
    """
    assert true.shape[0] == pred.shape[0]
    correct = np.equal(true, pred).sum()
    return float(correct) / float(true.shape[0])
