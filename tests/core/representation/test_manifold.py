from hover.core.representation.manifold import LayerwiseManifold
import numpy as np


def test_LayerwiseManifold(distance_preserving_array_sequence):
    LM = LayerwiseManifold(distance_preserving_array_sequence)
    LM.unfold(method="umap")
    _, disparities = LM.procrustes()
    assert (np.array(disparities) < 1e-16).all()
