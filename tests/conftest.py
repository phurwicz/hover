import pytest
import uuid
import re
import os
import numpy as np
import pandas as pd
import polars as pl
from functools import lru_cache

# configure hover
import hover

# hover.config.load_override(
#    random.choice(
#        [
#            os.path.join(os.path.dirname(__file__), _path)
#            for _path in [
#                "module_config/hover_alt_config_1.ini",
#                "module_config/hover_alt_config_2.ini",
#                "module_config/hover_alt_config_3.ini",
#            ]
#        ]
#    )
# )

from .local_config import (
    RANDOM_LABEL,
    RANDOM_SCORE,
    VECTORIZER_BREAKER,
)


def pytest_addoption(parser):
    parser.addoption(
        "--hover-ini",
        action="store",
        default="",
        help="Optional path to alternative hover config ini file",
    )


def pytest_configure(config):
    alt_config_path = os.path.join(os.getcwd(), config.getoption("--hover-ini"))
    if alt_config_path.endswith(".ini"):
        if os.path.isfile(alt_config_path):
            hover.config.load_override(alt_config_path)
        else:
            raise ValueError(f"Invalid config path {alt_config_path}")


@pytest.fixture(scope="module")
def dummy_vectorizer():
    from shaffle import uniform

    use_hashes = ["sha1", "sha224", "sha256", "sha384", "sha512", "md5"]

    @lru_cache(maxsize=int(1e5))
    def vectorizer(in_str):
        """
        Vectorizer with no semantic meaning but works for any string feature.
        """
        if in_str == VECTORIZER_BREAKER:
            raise ValueError("Special string made to break the test vectorizer")

        return np.array([uniform(in_str, _method) for _method in use_hashes])

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
        flag = re.search(r"(wheel|diesel|gasoline|automobile|vehicle)", row["text"])
        return "rec.autos" if flag else ABSTAIN_DECODED

    @labeling_function(targets=["rec.sport.baseball"])
    def baseball_keywords(row):
        flag = re.search(r"(baseball|stadium|\ bat\ |\ base\ )", row["text"])
        return "rec.sport.baseball" if flag else ABSTAIN_DECODED

    lf_list = [auto_keywords, baseball_keywords]

    return lf_list


@pytest.fixture(scope="module")
def generate_pandas_df_with_coords():
    import faker
    from hover.core.local_config import embedding_field
    from hover.module_config import ABSTAIN_DECODED

    fake_en = faker.Faker("en")

    def random_df_with_coords(size=300, embedding_dim=3):
        df = pd.DataFrame(
            [
                {
                    "text": fake_en.paragraph(3),
                    "audio": f"https://dom.ain/path/to/audio/file-{uuid.uuid1()}.mp3",
                    "image": f"https://dom.ain/path/to/image/file-{uuid.uuid1()}.jpg",
                    "label": ABSTAIN_DECODED,
                }
                for i in range(size)
            ]
        )
        for i in range(embedding_dim):
            _col = embedding_field(embedding_dim, i)
            df[_col] = np.random.normal(loc=0.0, scale=5.0, size=df.shape[0])
        return df

    return random_df_with_coords


def wrap_pandas_df(pandas_df):
    from hover.utils.dataframe import PandasDataframe, PolarsDataframe
    from hover.module_config import DataFrame

    assert isinstance(pandas_df, pd.DataFrame), f"Unexpected type {type(pandas_df)}"
    if DataFrame is PandasDataframe:
        return DataFrame(pandas_df)
    elif DataFrame is PolarsDataframe:
        return DataFrame(pl.from_pandas(pandas_df))
    else:
        raise ValueError(f"Unexpected DataFrame type {DataFrame}")


@pytest.fixture(scope="module")
def example_raw_pandas_df(generate_pandas_df_with_coords):
    return generate_pandas_df_with_coords(300)


@pytest.fixture(scope="module")
def example_raw_df(example_raw_pandas_df):
    return wrap_pandas_df(example_raw_pandas_df)


@pytest.fixture(scope="module")
def example_soft_label_df(generate_pandas_df_with_coords):
    df = generate_pandas_df_with_coords(100)
    df["pred_label"] = df.apply(RANDOM_LABEL, axis=1)
    df["pred_score"] = df.apply(RANDOM_SCORE, axis=1)
    return wrap_pandas_df(df)


@pytest.fixture(scope="module")
def example_margin_df(generate_pandas_df_with_coords):
    df = generate_pandas_df_with_coords(100)
    df["label_1"] = df.apply(RANDOM_LABEL, axis=1)
    df["label_2"] = df.apply(RANDOM_LABEL, axis=1)
    return wrap_pandas_df(df)


@pytest.fixture(scope="module")
def example_labeled_df(generate_pandas_df_with_coords):
    df = generate_pandas_df_with_coords(100)
    df["label"] = df.apply(RANDOM_LABEL, axis=1)
    return wrap_pandas_df(df)


@pytest.fixture(scope="module")
def example_everything_pandas_df(example_raw_pandas_df, generate_pandas_df_with_coords):
    from hover.core.local_config import DATASET_SUBSET_FIELD

    raw_df = example_raw_pandas_df.copy()
    raw_df[DATASET_SUBSET_FIELD] = "raw"
    labeled_df = generate_pandas_df_with_coords(200)
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


@pytest.fixture(scope="module")
def example_everything_df(example_everything_pandas_df):
    return wrap_pandas_df(example_everything_pandas_df)


def subroutine_dataset_with_vectorizer(pandas_df, dataset_cls, vectorizer):
    dataset = dataset_cls.from_pandas(pandas_df)
    dataset.vectorizer_lookup[2] = vectorizer
    return dataset


@pytest.fixture(scope="module")
def example_text_dataset(example_everything_pandas_df, dummy_vectorizer):
    from hover.core.dataset import SupervisableTextDataset

    dataset = subroutine_dataset_with_vectorizer(
        example_everything_pandas_df,
        SupervisableTextDataset,
        dummy_vectorizer,
    )

    return dataset


@pytest.fixture(scope="module")
def example_image_dataset(example_everything_pandas_df, dummy_vectorizer):
    from hover.core.dataset import SupervisableImageDataset

    return subroutine_dataset_with_vectorizer(
        example_everything_pandas_df,
        SupervisableImageDataset,
        dummy_vectorizer,
    )


@pytest.fixture(scope="module")
def example_audio_dataset(example_everything_pandas_df, dummy_vectorizer):
    from hover.core.dataset import SupervisableAudioDataset

    return subroutine_dataset_with_vectorizer(
        example_everything_pandas_df,
        SupervisableAudioDataset,
        dummy_vectorizer,
    )
