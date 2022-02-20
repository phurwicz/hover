import pytest
import pandas as pd
from hover import module_config
from hover.core.dataset import (
    SupervisableTextDataset,
    SupervisableImageDataset,
    SupervisableAudioDataset,
)
from hover.core.local_config import DATASET_SUBSET_FIELD
from .local_helper import (
    RANDOM_LABEL,
    RANDOM_SCORE,
)


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
def example_everything_df(example_raw_df, example_labeled_df):
    raw_df = example_raw_df.copy()
    raw_df[DATASET_SUBSET_FIELD] = "raw"
    labeled_df = example_labeled_df.copy()
    labeled_df[DATASET_SUBSET_FIELD] = "train"

    # should have these columns at this point:
    # - all features (text / image / audio)
    # - label, subset, and embeddings
    df = pd.concat([raw_df, labeled_df], axis=0)

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
