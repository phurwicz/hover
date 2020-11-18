"""
High-level workflows.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from hover.core.dataset import SupervisableTextDataset
from hover.core.neural import create_text_vector_net_from_module, TextVectorNet
from hover.utils.torch_helper import vector_dataloader, one_hot, label_smoothing
from wasabi import msg as logger


class VisualAnnotation:
    """
    Strongly coupled with the SupervisableDataset class.
    """

    def __init__(self, dataset, vectorizer=None, model_module_name=None):
        """
        """
        assert isinstance(dataset, SupervisableDataset)
        self.dataset = dataset
        self.model_module_name = model_module_name

    def compute_text_to_2d(self, method, **kwargs):
        """
        Calculate a 2D manifold, excluding the test set.
        :param method: the dimensionality reduction method to use.
        :type method: str, "umap" or "ivis"
        """
        from hover.representation.reduction import DimensionalityReducer

        vectorizer = create_text_vector_net_from_module(
            TextVectorNet, self.model_module_name, self.dataset.classes
        ).vectorizer

        # prepare input vectors to manifold learning
        subset = ["raw", "train", "dev"]
        fit_texts = []
        for _key in subset:
            _df = self.dataset.dfs[_key]
            if _df.empty:
                continue
            fit_texts += _df["text"].tolist()
        fit_arr = np.array([vectorizer(_text) for _text in tqdm(fit_texts)])

        # initialize and fit manifold learning reducer
        reducer = DimensionalityReducer(fit_arr)
        embedding = reducer.fit_transform(method, **kwargs)

        # assign x and y coordinates to dataset
        start_idx = 0
        for _key in subset:
            _df = self.dataset.dfs[_key]
            _length = _df.shape[0]
            _df["x"] = pd.Series(embedding[start_idx : (start_idx + _length), 0])
            _df["y"] = pd.Series(embedding[start_idx : (start_idx + _length), 1])
            start_idx += _length

    def get_loader(self, key, vectorizer, batch_size=64, smoothing_coeff=0.0):
        """
        Prepare a Torch Dataloader for training or evaluation.
        :param key: the subset of dataset to use.
        :type key: str
        :param vectorizer: callable that turns a string into a vector.
        :type vectorizer: callable
        :param smoothing_coeff: the smoothing coeffient for soft labels.
        :type smoothing_coeff: float
        """
        labels = (
            self.dataset.dfs[key]["label"]
            .apply(lambda x: self.dataset.label_encoder[x])
            .tolist()
        )
        texts = self.dataset.dfs[key]["text"].tolist()
        output_vectors = one_hot(labels, num_classes=len(self.dataset.classes))

        logger.info(f"Preparing input vectors...")
        input_vectors = [vectorizer(_text) for _text in tqdm(texts)]
        output_vectors = label_smoothing(
            output_vectors,
            num_classes=len(self.dataset.classes),
            coefficient=smoothing_coeff,
        )
        logger.info(f"Preparing data loader...")
        loader = vector_dataloader(input_vectors, output_vectors, batch_size=batch_size)
        logger.good(
            f"Prepared {key} loader consisting of {len(texts)} examples with batch size {batch_size}"
        )
        return loader

    def model_from_dev(self, **kwargs):
        """
        Train a Prodigy-compatible model from the dev set.
        """
        model = create_text_vector_net_from_module(
            TextVectorNet, self.model_module_name, self.dataset.classes
        )
        dev_loader = self.get_loader("dev", model.vectorizer, smoothing_coeff=0.1)
        train_info = model.train(dev_loader, dev_loader, **kwargs)
        return model, train_info
