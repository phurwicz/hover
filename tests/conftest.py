import pytest
import random
import uuid
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from hover import module_config
from hover.core.dataset import (
    SupervisableTextDataset,
    SupervisableImageDataset,
    SupervisableAudioDataset,
)
from hover.core.local_config import (
    embedding_field,
    DATASET_SUBSET_FIELD,
)
from .local_config import (
    RANDOM_LABEL,
    RANDOM_SCORE,
    VECTORIZER_BREAKER,
)


@pytest.fixture(scope="module")
def dummy_vectorizer():
    from hashlib import sha1, sha224, sha256, sha384, sha512, md5

    use_hashes = [sha1, sha224, sha256, sha384, sha512, md5]
    max_plus_ones = [2 ** (_hash().digest_size * 8) for _hash in use_hashes]

    @lru_cache(maxsize=int(1e5))
    def vectorizer(in_str):
        """
        Vectorizer with no semantic meaning but works for any string feature.
        """
        if in_str == VECTORIZER_BREAKER:
            raise ValueError("Special string made to break the test vectorizer")

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
    import faker

    fake_en = faker.Faker("en")

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
def example_raw_df(generate_df_with_coords):
    df = generate_df_with_coords(300)
    df["label"] = module_config.ABSTAIN_DECODED
    return df


@pytest.fixture(scope="module")
def example_soft_label_df(example_raw_df):
    df = example_raw_df.copy()
    df["pred_label"] = df.apply(RANDOM_LABEL, axis=1)
    df["pred_score"] = df.apply(RANDOM_SCORE, axis=1)
    return df


@pytest.fixture(scope="module")
def example_margin_df(example_raw_df):
    df = example_raw_df.copy()
    df["label_1"] = df.apply(RANDOM_LABEL, axis=1)
    df["label_2"] = df.apply(RANDOM_LABEL, axis=1)
    return df


@pytest.fixture(scope="module")
def example_labeled_df(generate_df_with_coords):
    df = generate_df_with_coords(100)
    df["label"] = df.apply(RANDOM_LABEL, axis=1)
    return df


@pytest.fixture(scope="module")
def example_everything_df(example_raw_df, generate_df_with_coords):
    raw_df = example_raw_df.copy()
    raw_df[DATASET_SUBSET_FIELD] = "raw"
    labeled_df = generate_df_with_coords(200)
    labeled_df["label"] = labeled_df.apply(RANDOM_LABEL, axis=1)
    labeled_df[DATASET_SUBSET_FIELD] = "train"
    labeled_df.loc[100:150, DATASET_SUBSET_FIELD] = "dev"
    labeled_df.loc[150:, DATASET_SUBSET_FIELD] = "test"

    # should have these columns at this point:
    # - all features (text / image / audio)
    # - label, subset, and embeddings
    df = pd.concat([raw_df, labeled_df], axis=0).reset_index(drop=True)

    # prepare columns for special functionalities
    df["pred_label"] = df.apply(RANDOM_LABEL, axis=1)
    df["pred_score"] = df.apply(RANDOM_SCORE, axis=1)
    df["label_1"] = df.apply(RANDOM_LABEL, axis=1)
    df["label_2"] = df.apply(RANDOM_LABEL, axis=1)
    return df


def subroutine_dataset_with_vectorizer(df, dataset_cls, vectorizer):
    dataset = dataset_cls.from_pandas(df)
    dataset.vectorizer_lookup[2] = vectorizer
    return dataset


@pytest.fixture(scope="module")
def example_text_dataset(example_everything_df, dummy_vectorizer):
    return subroutine_dataset_with_vectorizer(
        example_everything_df,
        SupervisableTextDataset,
        dummy_vectorizer,
    )


@pytest.fixture(scope="module")
def example_image_dataset(example_everything_df, dummy_vectorizer):
    return subroutine_dataset_with_vectorizer(
        example_everything_df,
        SupervisableImageDataset,
        dummy_vectorizer,
    )


@pytest.fixture(scope="module")
def example_audio_dataset(example_everything_df, dummy_vectorizer):
    return subroutine_dataset_with_vectorizer(
        example_everything_df,
        SupervisableAudioDataset,
        dummy_vectorizer,
    )
