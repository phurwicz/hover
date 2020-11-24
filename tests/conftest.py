import pytest
import random


@pytest.fixture(scope="module")
def mini_df_text():
    from hover.utils.datasets import newsgroups_dictl
    import pandas as pd

    dataset, _, _ = newsgroups_dictl()

    mini_dictl = random.sample(dataset["train"], k=1000)
    mini_df = pd.DataFrame(mini_dictl)

    return mini_df
