"""
Dataset objects which extend beyond DataFrames.
"""
from abc import ABC, abstractmethod
from rich.console import Console
import pandas as pd
import numpy as np
import wrappy
from hover import module_config
from tqdm import tqdm

console = Console()


class SupervisableDataset(ABC):
    """
    Type-agnostic class for a dataset open to supervision.
    Raw -- piecewise annoatation -> Gold -> Dev/Test
    Raw -- batch annotation -> Noisy -> Train

    Keeping a DataFrame form and a list-of-dicts form, with the intention
    that synchronization should be called manually and sparingly.
    """

    ORDERED_SUBSET = ("test", "dev", "train", "raw")
    FEATURE_KEY = "feature"

    def __init__(
        self,
        raw_dictl,
        train_dictl=[],
        dev_dictl=[],
        test_dictl=[],
        feature_key="feature",
        label_key="label",
    ):
        """
        Initialize the dataset with dictl and df forms; initialize the mapping between categorical-int and string labels.
        :param raw_dictl: a list of dicts holding the raw data that DO NOT have annotation.
        :param train_dictl: a list of dicts holding the batch-annotated noisy train set.
        :param dev_dictl: a list of dicts holding the gold dev set.
        :param test_dictl: a list of dicts holding the gold test set.
        :param feature_key: key in each piece of dict mapping to the feature.
        :param label_key: key in each piece of dict mapping to the ground truth in STRING form.
        """

        def dictl_transform(dictl, labels=True):
            """
            Burner function to transform the input list of dictionaries into standard format.
            """
            key_transform = {feature_key: self.__class__.FEATURE_KEY}
            if labels:
                key_transform[label_key] = "label"
            return [
                {
                    key_transform.get(_key, _key): _value
                    for _key, _value in _dict.items()
                }
                for _dict in dictl
            ]

        self.dictls = {
            "raw": dictl_transform(raw_dictl, labels=False),
            "train": dictl_transform(train_dictl),
            "dev": dictl_transform(dev_dictl),
            "test": dictl_transform(test_dictl),
        }

        self.synchronize_dictl_to_df()
        self.df_deduplicate()
        self.synchronize_df_to_dictl()
        self.setup_label_coding()

    def setup_label_coding(self):
        """
        Auto-determine labels in the dataset, then create encoder/decoder in lexical order.
        Add ABSTAIN as a no-label placeholder.
        """
        all_labels = set()
        for _key in self.__class__.ORDERED_SUBSET[:-1]:
            _df = self.dfs[_key]
            if _df.empty:
                continue
            assert "label" in _df.columns
            _found_labels = set(_df["label"].tolist())
            all_labels = all_labels.union(_found_labels)

        self.classes = sorted(all_labels)
        self.label_encoder = {
            **{_label: _i for _i, _label in enumerate(self.classes)},
            module_config.ABSTAIN_DECODED: module_config.ABSTAIN_ENCODED,
        }
        self.label_decoder = {_v: _k for _k, _v in self.label_encoder.items()}

        console.print(
            f"Set up label encoder/decoder with {len(self.classes)} classes.",
            style="green",
        )
        self.validate_labels()

    def validate_labels(self, raise_exception=True):
        """
        Check that every label is in the encoder.
        """
        for _key in self.__class__.ORDERED_SUBSET[:-1]:
            _df = self.dfs[_key]
            _invalid_indices = None
            if _df.empty:
                continue
            assert "label" in _df.columns
            _mask = _df["label"].apply(lambda x: x in self.label_encoder)
            _invalid_indices = np.where(_mask == False)[0].tolist()
            if _invalid_indices:
                console.print(f"Subset [{_key}] has invalid labels:")
                console.print({_df.loc[_invalid_indices]})
                if raise_exception:
                    raise ValueError("invalid labels")

    def df_deduplicate(self):
        """
        Cross-deduplicate data entries by feature between subsets.
        """
        # keep track of which df has which columns and which rows came from which subset
        columns = dict()
        for _key in self.__class__.ORDERED_SUBSET:
            console.print(
                f"--subset {_key} has {self.dfs[_key].shape[0]} entries before.",
                style="blue",
            )
            columns[_key] = self.dfs[_key].columns
            self.dfs[_key]["__subset"] = _key

        # concatenate in order and deduplicate
        overall_df = pd.concat(
            [self.dfs[_key] for _key in self.__class__.ORDERED_SUBSET],
            axis=0,
            sort=False,
        )
        overall_df.drop_duplicates(
            subset=[self.__class__.FEATURE_KEY], keep="first", inplace=True
        )
        overall_df.reset_index(drop=True, inplace=True)

        # cut up slices
        for _key in self.__class__.ORDERED_SUBSET:
            self.dfs[_key] = overall_df[overall_df["__subset"] == _key].reset_index(
                drop=True, inplace=False
            )[columns[_key]]
            console.print(
                f"--subset {_key} has {self.dfs[_key].shape[0]} entries after.",
                style="blue",
            )

    def synchronize_dictl_to_df(self):
        """
        Re-make dataframes from lists of dictionaries.
        """
        self.dfs = dict()
        for _key, _dictl in self.dictls.items():
            self.dfs[_key] = pd.DataFrame(_dictl)

    def synchronize_df_to_dictl(self):
        """
        Re-make lists of dictionaries from dataframes.
        """
        self.dictls = dict()
        for _key, _df in self.dfs.items():
            self.dictls[_key] = _df.to_dict(orient="records")

    def compute_2d_embedding(self, vectorizer, method, **kwargs):
        """
        Get embeddings in the xy-plane and return the reducer.
        """
        from hover.core.representation.reduction import DimensionalityReducer

        # prepare input vectors to manifold learning
        subset = ["raw", "train", "dev"]
        fit_inp = []
        for _key in subset:
            _df = self.dfs[_key]
            if _df.empty:
                continue
            fit_inp += _df[self.__class__.FEATURE_KEY].tolist()
        fit_arr = np.array([vectorizer(_inp) for _inp in tqdm(fit_inp)])

        # initialize and fit manifold learning reducer
        reducer = DimensionalityReducer(fit_arr)
        embedding = reducer.fit_transform(method, **kwargs)

        # assign x and y coordinates to dataset
        start_idx = 0
        for _key in subset:
            _df = self.dfs[_key]
            _length = _df.shape[0]
            _df["x"] = pd.Series(embedding[start_idx : (start_idx + _length), 0])
            _df["y"] = pd.Series(embedding[start_idx : (start_idx + _length), 1])
            start_idx += _length

        return reducer


class SupervisableTextDataset(SupervisableDataset):
    """
    Can add text-specific methods.
    """

    FEATURE_KEY = "text"

    pass
