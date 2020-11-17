"""
Dataset objects which extend beyond DataFrames.
"""
from abc import ABC, abstractmethod
from rich.console import Console
import pandas as pd
import numpy as np
import wrappy

console = Console()


class SupervisableDataset(ABC):
    """
    Type-agnostic class for a dataset open to supervision.
    """

    pass


class SupervisableTextDataset(SupervisableDataset):
    """
    Raw -- piecewise annoatation -> Gold -> Dev/Test
    Raw -- batch annotation -> Noisy -> Train

    Keeping a DataFrame form and a list-of-dicts form, with the intention
    that synchronization should be called manually and sparingly.
    """

    ORDERED_SUBSET = ("test", "dev", "train", "raw")

    def __init__(
        self,
        raw_dictl,
        train_dictl=[],
        dev_dictl=[],
        test_dictl=[],
        text_key="text",
        label_key="label",
    ):
        """
        Initialize the dataset with dictl and df forms; initialize the mapping between categorical-int and string labels.
        :param raw_dictl: a list of dicts holding the raw data that DO NOT have annotation.
        :param train_dictl: a list of dicts holding the batch-annotated noisy train set.
        :param dev_dictl: a list of dicts holding the gold dev set.
        :param test_dictl: a list of dicts holding the gold test set.
        :param text_key: key in each piece of dict mapping to the text.
        :param label_key: key in each piece of dict mapping to the ground truth in STRING form.
        """

        def dictl_transform(dictl, labels=True):
            """
            Burner function to transform the input list of dictionaries into standard format.
            """
            key_transform = {text_key: "text"}
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
            "raw": dictl_transform(raw_dictl),
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
        Note: think about ABSTAIN -- should it be allowed as a label in the raw/train set?
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
        self.label_encoder = {_label: _i for _i, _label in enumerate(self.classes)}
        self.label_decoder = {_i: _label for _i, _label in enumerate(self.classes)}

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
        Cross-deduplicate data entries by text between subsets.
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
        overall_df.drop_duplicates(subset=["text"], keep="first", inplace=True)
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
