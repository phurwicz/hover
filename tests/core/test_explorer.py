"""
Note that the whole point of explorers is to allow interaction, for which this file should not be considered a complete suite of tests.
"""

from hover.core.explorer import (
    BokehCorpusExplorer,
    BokehCorpusAnnotator,
    BokehMarginExplorer,
    BokehSnorkelExplorer,
)
import pytest
import pandas as pd
import faker
import random

fake_en = faker.Faker("en")

EXAMPLE_RAW_DF = pd.DataFrame(
    [
        {
            "text": fake_en.paragraph(3),
            "x": random.uniform(-1.0, 1.0),
            "y": random.uniform(-1.0, 1.0),
        }
        for i in range(300)
    ]
)

EXAMPLE_MARGIN_DF = EXAMPLE_RAW_DF.copy()
EXAMPLE_MARGIN_DF["label_1"] = EXAMPLE_MARGIN_DF["text"].apply(
    lambda x: random.choice(["A", "B"])
)
EXAMPLE_MARGIN_DF["label_2"] = EXAMPLE_MARGIN_DF["text"].apply(
    lambda x: random.choice(["A", "B"])
)

EXAMPLE_DEV_DF = pd.DataFrame(
    [
        {
            "text": fake_en.paragraph(3),
            "x": random.uniform(-1.0, 1.0),
            "y": random.uniform(-1.0, 1.0),
            "label": random.choice(["A", "B"]),
        }
        for i in range(100)
    ]
)


@pytest.mark.core
class TestBokehCorpusExplorer:
    @staticmethod
    def test_init():
        explorer = BokehCorpusExplorer({"raw": EXAMPLE_RAW_DF})
        _ = explorer.view()


@pytest.mark.core
class TestBokehCorpusAnnotator:
    @staticmethod
    def test_init():
        explorer = BokehCorpusAnnotator({"raw": EXAMPLE_RAW_DF})
        _ = explorer.view()


@pytest.mark.core
class TestBokehMarginExplorer:
    @staticmethod
    def test_init():
        explorer = BokehMarginExplorer({"raw": EXAMPLE_MARGIN_DF}, "label_1", "label_2")
        explorer.plot("A")
        explorer.plot("B")
        _ = explorer.view()


@pytest.mark.core
class TestBokehSnorkelExplorer:
    @staticmethod
    def test_init():
        explorer = BokehSnorkelExplorer(
            {"raw": EXAMPLE_RAW_DF, "labeled": EXAMPLE_DEV_DF}
        )
        _ = explorer.view()
