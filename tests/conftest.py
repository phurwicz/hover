import pytest
import random
import spacy
import faker
import re
import pandas as pd
from hover.utils.datasets import newsgroups_dictl, newsgroups_reduced_dictl
from hover.core.dataset import SupervisableTextDataset

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
def dummy_vecnet_callback():
    from hover.core.neural import VectorNet
    from hover.utils.common_nn import LogisticRegression

    def callback(dataset, vectorizer):
        vecnet = VectorNet(
            vectorizer, LogisticRegression, ".model.test.pt", dataset.classes
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
                    "audio": "https://www.soundjay.com/button/beep-01a.mp3",
                    "image": "https://docs.chainer.org/en/stable/_images/5.png",
                    "x": random.uniform(-1.0, 1.0),
                    "y": random.uniform(-1.0, 1.0),
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
    dataset.synchronize_df_to_dictl()

    return dataset
