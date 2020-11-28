import pytest
import random
import spacy
import faker
import re
import pandas as pd
from hover.utils.datasets import newsgroups_dictl, newsgroups_reduced_dictl
from hover.core.dataset import SupervisableTextDataset
from copy import deepcopy

fake_en = faker.Faker("en")


@pytest.fixture(scope="module")
def spacy_en_md():
    nlp = spacy.load("en_core_web_md")
    return nlp


@pytest.fixture(scope="module")
def dummy_vectorizer(spacy_en_md):
    def vectorizer(text):
        clean_text = re.sub(r"[\t\n]", r" ", text)
        to_disable = spacy_en_md.pipe_names
        doc = spacy_en_md(clean_text, disable=to_disable)
        return doc.vector

    trial_vector = vectorizer("hi")
    assert trial_vector.shape == (300,)

    return vectorizer


@pytest.fixture(scope="module")
def generate_text_df_with_coords():
    def random_text_df_with_coords(size=300):
        df = pd.DataFrame(
            [
                {
                    "text": fake_en.paragraph(3),
                    "x": random.uniform(-1.0, 1.0),
                    "y": random.uniform(-1.0, 1.0),
                }
                for i in range(300)
            ]
        )
        return df

    return random_text_df_with_coords


@pytest.fixture(scope="module")
def mini_df_text():
    my_20ng, _, _ = newsgroups_reduced_dictl()

    mini_dictl = random.sample(my_20ng["train"], k=1000)
    mini_df = pd.DataFrame(mini_dictl)

    return mini_df


@pytest.fixture(scope="module")
def mini_supervisable_text_dataset():
    my_20ng, _, _ = newsgroups_dictl()

    split_idx = int(0.05 * len(my_20ng["train"]))
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
