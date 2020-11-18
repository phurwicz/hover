from hover.utils.metrics import classification_accuracy
import numpy as np

def test_classification_accuracy():
    true = np.array([1, 2, 3, 4, 5, 6, 7, 7])
    pred = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    accl = classification_accuracy(true, pred)
    accr = classification_accuracy(pred, true)
    assert np.allclose(accl, 7/8)
    assert np.allclose(accr, 7/8)
