import pytest
import random
import spacy
import faker
import uuid
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from hover.utils.datasets import newsgroups_dictl, newsgroups_reduced_dictl
from hover.core.dataset import SupervisableTextDataset
from hover.core.local_config import embedding_field

fake_en = faker.Faker("en")


@pytest.fixture(scope="module")
def spacy_en_md():
    nlp = spacy.load("en_core_web_md")
    return nlp


@pytest.fixture(scope="module")
def dummy_vectorizer(spacy_en_md):
    from hashlib import sha1, sha224, sha256, sha384, sha512, md5

    use_hashes = [sha1, sha224, sha256, sha384, sha512, md5]
    max_plus_ones = [2 ** (_hash().digest_size * 8) for _hash in use_hashes]

    @lru_cache(maxsize=int(1e5))
    def vectorizer(in_str):
        """
        Vectorizer with no semantic meaning but works for any string feature.
        """
        arr = []
        seed = in_str.encode()
        for _hash, _max_plus_one in zip(use_hashes, max_plus_ones):
            _hash_digest = _hash(seed).digest()
            _hash_int = int.from_bytes(_hash_digest, "big")
            arr.append(_hash_int / _max_plus_one)
        return np.array(arr)

    return vectorizer


@pytest.fixture(scope="module")
def dummy_vecnet_callback(dummy_vectorizer):
    from hover.core.neural import VectorNet
    from hover.utils.common_nn import LogisticRegression

    def callback(dataset):
        vecnet = VectorNet(
            dummy_vectorizer,
            LogisticRegression,
            f".model.test.{uuid.uuid1()}.pt",
            dataset.classes,
        )
        return vecnet

    return callback


@pytest.fixture(scope="module")
def dummy_labeling_function_list():
    from hover.utils.snorkel_helper import labeling_function
    from hover.module_config import ABSTAIN_DECODED

    @labeling_function(targets=["rec.autos"])
    def auto_keywords(row):
        flag = re.search(r"(wheel|diesel|gasoline|automobile|vehicle)", row.text)
        return "rec.autos" if flag else ABSTAIN_DECODED

    @labeling_function(targets=["rec.sport.baseball"])
    def baseball_keywords(row):
        flag = re.search(r"(baseball|stadium|\ bat\ |\ base\ )", row.text)
        return "rec.sport.baseball" if flag else ABSTAIN_DECODED

    lf_list = [auto_keywords, baseball_keywords]

    return lf_list


@pytest.fixture(scope="module")
def generate_df_with_coords():
    def random_df_with_coords(size=300):
        df = pd.DataFrame(
            [
                {
                    "text": fake_en.paragraph(3),
                    "audio": f"https://dom.ain/path/to/audio/file-{uuid.uuid1()}.mp3",
                    "image": f"https://dom.ain/path/to/image/file-{uuid.uuid1()}.jpg",
                    embedding_field(3, 0): random.uniform(-1.0, 1.0),
                    embedding_field(3, 1): random.uniform(-1.0, 1.0),
                    embedding_field(3, 2): random.uniform(-1.0, 1.0),
                }
                for i in range(size)
            ]
        )
        return df

    return random_df_with_coords


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
        train_dictl=my_20ng["train"][split_idx : int(split_idx * 1.5)],
        dev_dictl=my_20ng["train"][int(split_idx * 1.5) : int(split_idx * 1.7)],
        test_dictl=my_20ng["test"][: int(split_idx * 0.2)],
    )

    return dataset


@pytest.fixture(scope="module")
def mini_supervisable_text_dataset_embedded(
    mini_supervisable_text_dataset, dummy_vectorizer
):
    dataset = mini_supervisable_text_dataset.copy()

    dataset.compute_2d_embedding(dummy_vectorizer, "umap")
    return dataset


@pytest.fixture(scope="module")
def noisy_supervisable_text_dataset():
    my_20ng, _, _ = newsgroups_dictl()

    split_idx = int(0.8 * len(my_20ng["train"]))
    dataset = SupervisableTextDataset(
        raw_dictl=[],
        train_dictl=my_20ng["train"][:split_idx],
        dev_dictl=my_20ng["train"][split_idx:],
        test_dictl=my_20ng["test"],
    )

    def mutate(value, prob=0.5, pool=dataset.classes):
        """
        Up to prob, return a draw from the pool instead of the original value.
        """
        if random.uniform(0.0, 1.0) < prob:
            return random.sample(pool, 1)[0]
        return value

    dataset.dfs["train"]["label"] = dataset.dfs["train"]["label"].apply(mutate)
    return dataset
