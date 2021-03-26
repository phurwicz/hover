import pytest
from hover.core.dataset import SupervisableTextDataset


@pytest.mark.core
class TestSupervisableTextDataset:
    TEST_DICTL = [
        {"content": "Aristotle", "mark": "A"},
        {"content": "Bertrand Russell", "mark": "B"},
        {"content": "CRISPR", "mark": "C"},
        {"content": "Doudna", "mark": "D"},
    ]
    DEV_DICTL = [
        {"content": "Doudna", "mark": "D"},
        {"content": "Ernest Hemingway", "mark": "E"},
    ]
    TRAIN_DICTL = [
        {"content": "Ernest Hemingway", "mark": "e"},
        {"content": "Franklin Roosevelt", "mark": "F"},
        {"content": "Geralt of Rivia", "mark": "G"},
    ]
    RAW_DICTL = [{"content": "Geralt of Rivia"}, {"content": "Hailstorm"}]

    EFFECTIVE_SIZE = {"test": 4, "dev": 1, "train": 2, "raw": 1}

    EFFECTIVE_CLASSES = 7

    def test_init(self):
        dataset = SupervisableTextDataset(
            self.__class__.RAW_DICTL[:],
            train_dictl=self.__class__.TRAIN_DICTL[:],
            dev_dictl=self.__class__.DEV_DICTL[:],
            test_dictl=self.__class__.TEST_DICTL[:],
            feature_key="content",
            label_key="mark",
        )

        # check the subset sizes
        for _key, _value in self.__class__.EFFECTIVE_SIZE.items():
            assert dataset.dfs[_key].shape[0] == _value

        # check the number of classes
        assert len(dataset.classes) == self.__class__.EFFECTIVE_CLASSES

    @staticmethod
    def test_export_import(mini_supervisable_text_dataset):
        dataset = mini_supervisable_text_dataset

        df = dataset.to_pandas(use_df=True)
        df = dataset.to_pandas(use_df=False)
        dataset = SupervisableTextDataset.from_pandas(df)

    @staticmethod
    def test_compute_2d_embedding(mini_supervisable_text_dataset, dummy_vectorizer):
        dataset = mini_supervisable_text_dataset

        dataset.compute_2d_embedding(dummy_vectorizer, "umap")
        dataset.compute_2d_embedding(dummy_vectorizer, "ivis")

    @staticmethod
    def test_loader(mini_supervisable_text_dataset, dummy_vectorizer):
        from torch.utils.data import DataLoader

        dataset = mini_supervisable_text_dataset

        try:
            loader = dataset.loader("raw", dummy_vectorizer, smoothing_coeff=0.1)
            pytest.fail(
                "The raw subset managed to produce a loader, which should not happen"
            )
        except ValueError:
            loader = dataset.loader("dev", dummy_vectorizer, smoothing_coeff=0.1)
            assert isinstance(loader, DataLoader)
