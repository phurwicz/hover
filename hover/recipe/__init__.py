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
from wrappy import todo


class VisualAnnotation:
    """
    """

    @todo(
        "Think about what belongs in this class and what belongs in the core modules."
    )
    def __init__(self, dataset, vectorizer=None, model_module_name=None):
        """
        """
        assert isinstance(dataset, SupervisableDataset)
        self.dataset = dataset
        self.model_module_name = model_module_name

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
