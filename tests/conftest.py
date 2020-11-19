import pytest


@pytest.fixture(scope="module")
def mini_df_text():
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_test = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"), categories=["alt.atheism", "sci.space"]
    )

    return newsgroups_test
