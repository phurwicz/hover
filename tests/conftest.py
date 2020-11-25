import pytest
import random
import spacy
import re
import numpy as np
import pandas as pd
from hover.utils.datasets import newsgroups_dictl
from hover.core.dataset import SupervisableTextDataset
from copy import deepcopy


@pytest.fixture(scope="module")
def tiny_spacy():
    nlp = spacy.load("en_core_web_sm")
    return nlp


@pytest.fixture(scope="module")
def dummy_vectorizer(tiny_spacy):
    def vectorizer(text):
        clean_text = re.sub(r"[\t\n]", r" ", text)
        doc = tiny_spacy(clean_text, disable=nlp.pipe_names)
        return doc.vector

    return vectorizer


@pytest.fixture(scope="module")
def mini_df_text():
    my_20ng, _, _ = newsgroups_dictl()

    mini_dictl = random.sample(my_20ng["train"], k=1000)
    mini_df = pd.DataFrame(mini_dictl)

    return mini_df


@pytest.fixture(scope="module")
def mini_supervisable_text_dataset():
    my_20ng, _, _ = newsgroups_dictl()

    split_idx = int(0.2 * len(my_20ng["train"]))
    dataset = SupervisableTextDataset(
        raw_dictl=my_20ng["train"][:split_idx],
        dev_dictl=my_20ng["train"][split_idx : int(split_idx * 1.2)],
        test_dictl=my_20ng["test"][:split_idx],
    )

    dataset.dfs["raw"].drop(["label"], inplace=True, axis=1)
    dataset.synchronize_df_to_dictl()

    return dataset


@pytest.fixture(scope="module")
def mini_supervisable_text_dataset_embedded(
    mini_supervisable_text_dataset, dummy_vectorizer
):
    dataset = deepcopy(mini_supervisable_text_dataset)

    dataset.compute_2d_embedding(dummy_vectorizer, "umap")

    return dataset
