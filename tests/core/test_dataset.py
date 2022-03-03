import pytest
import os
from hover.core.dataset import SupervisableTextDataset
from bokeh.events import MenuItemClick


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

    @pytest.mark.lite
    def test_init(self):
        dataset = SupervisableTextDataset(
            self.__class__.RAW_DICTL[:],
            train_dictl=self.__class__.TRAIN_DICTL[:],
            dev_dictl=self.__class__.DEV_DICTL[:],
            test_dictl=self.__class__.TEST_DICTL[:],
            feature_key="content",
            label_key="mark",
        )
        dataset.validate_labels()

        # check the subset sizes
        for _key, _value in self.__class__.EFFECTIVE_SIZE.items():
            assert dataset.dfs[_key].shape[0] == _value

        # check the number of classes
        assert len(dataset.classes) == self.__class__.EFFECTIVE_CLASSES

    @staticmethod
    @pytest.mark.lite
    def test_basic_logging(example_text_dataset):
        dataset = example_text_dataset.copy()
        dataset._print("testing _print(); not expecting color")
        dataset._good("testing _good(); expecting green")
        dataset._info("testing _info(); expecting blue")
        dataset._warn("testing _warn(); expecting yellow")
        dataset._fail("testing _fail(); expecting red")

    @staticmethod
    @pytest.mark.lite
    def test_setup_label_coding(example_text_dataset):
        dataset = example_text_dataset.copy()
        dataset.setup_label_coding(verbose=True, debug=True)

        assert len(dataset.classes) + 1 == len(dataset.label_encoder)
        assert len(dataset.label_encoder) == len(dataset.label_decoder)

    @staticmethod
    @pytest.mark.lite
    def test_validate_labels(example_text_dataset):
        dataset = example_text_dataset.copy()
        dataset.dfs["train"].at[0, "label"] = "invalid_label"

        try:
            dataset.validate_labels()
            pytest.fail("Expected exception caused by label uncaught by encoder.")
        except ValueError:
            pass

        dataset.validate_labels(raise_exception=False)

    @staticmethod
    @pytest.mark.lite
    def test_compute_feature_index(example_text_dataset):
        dataset = example_text_dataset.copy()
        dataset.dfs["raw"].at[0, "text"] = dataset.dfs["raw"].at[1, "text"]

        try:
            dataset.compute_feature_index()
            pytest.fail("Expected exception caused by duplicate feature.")
        except ValueError:
            pass

    @staticmethod
    @pytest.mark.lite
    def test_locate_by_feature_value(example_text_dataset):
        dataset = example_text_dataset.copy()
        feature_value = dataset.dfs["raw"].at[0, "text"]
        subset, index = dataset.locate_by_feature_value(feature_value)
        assert subset == "raw" and index == 0

        dataset.feature_to_subset_idx[feature_value] = ("raw", 1)

        try:
            dataset.locate_by_feature_value(feature_value, auto_recompute=False)
            pytest.fail("Expected exception caused by inconsistent feature index.")
        except ValueError:
            # auto-recompute should restore consistency
            dataset.locate_by_feature_value(feature_value)

    @staticmethod
    @pytest.mark.lite
    def test_export_import(example_text_dataset):
        dataset = example_text_dataset.copy()

        df = dataset.to_pandas()
        dataset = SupervisableTextDataset.from_pandas(df)

        # trigger callback through button click (UI behavior)
        for _item, _ext in [
            ("Excel", ".xlsx"),
            ("CSV", ".csv"),
            ("JSON", ".json"),
            ("pickle", ".pkl"),
        ]:
            _f_old = [_path for _path in os.listdir(".") if _path.endswith(_ext)]
            _event = MenuItemClick(dataset.file_exporter, item=_item)
            dataset.file_exporter._trigger_event(_event)
            _f_new = [_path for _path in os.listdir(".") if _path.endswith(_ext)]
            assert len(_f_new) == len(_f_old) + 1

    @staticmethod
    @pytest.mark.lite
    def test_compute_nd_embedding(example_text_dataset, dummy_vectorizer):
        dataset = example_text_dataset.copy()

        dataset.compute_nd_embedding(dummy_vectorizer, "umap", dimension=3)

        # empty one of the dfs; should not break the method
        dataset.dfs["test"] = dataset.dfs["test"].loc[0:0]
        dataset.compute_2d_embedding(dummy_vectorizer, "umap")

        # verify that the vectorizer has been remembered
        assert len(dataset.vectorizer_lookup) > 0

    @staticmethod
    @pytest.mark.lite
    def test_vectorizer_lookup(example_text_dataset):
        dataset = example_text_dataset.copy()
        to_assign = dict()
        # this assignment should get prevented
        dataset.vectorizer_lookup = to_assign
        assert dataset.vectorizer_lookup is not to_assign

    @staticmethod
    @pytest.mark.lite
    def test_loader(example_text_dataset, dummy_vectorizer):
        from torch.utils.data import DataLoader

        dataset = example_text_dataset.copy()

        try:
            loader = dataset.loader("raw", dummy_vectorizer, smoothing_coeff=0.1)
            pytest.fail(
                "The raw subset managed to produce a loader, which should not happen"
            )
        except ValueError:
            # single vectorizer
            loader = dataset.loader("dev", dummy_vectorizer, smoothing_coeff=0.1)
            assert isinstance(loader, DataLoader)
            # multiple vectorizers
            loader = dataset.loader(
                "dev", dummy_vectorizer, dummy_vectorizer, smoothing_coeff=0.1
            )
            assert isinstance(loader, DataLoader)
