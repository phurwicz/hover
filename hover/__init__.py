"""
Module root where constants get configured.
"""
from flexmod import AutolockedConfigValue, Config, ConfigIndex
from bokeh.palettes import Turbo256

config = ConfigIndex(
    [
        Config(
            "io",
            [
                AutolockedConfigValue(
                    "data_save_dir",
                    "The directory path for saving labeled data.",
                    ".",
                ),
            ],
        ),
        Config(
            "visual",
            [
                AutolockedConfigValue(
                    "abstain_hexcolor",
                    "Hex code of RGB color.",
                    "#dcdcdc",
                ),
                AutolockedConfigValue(
                    "bokeh_palette",
                    "The bokeh color palette to use for plotting.",
                    Turbo256,
                ),
            ],
        ),
        Config(
            "data.embedding",
            [
                AutolockedConfigValue(
                    "default_reduction_method",
                    "Default method for dimensionality reduction. Currently either 'umap' or 'ivis'.",
                    "umap",
                ),
            ],
        ),
        Config(
            "data.columns",
            [
                AutolockedConfigValue(
                    "encoded_label_key",
                    "The column name for the encoded label.",
                    "label_encoded",
                ),
                AutolockedConfigValue(
                    "dataset_subset_field",
                    "The column name for dataset subsets.",
                    "SUBSET",
                ),
                AutolockedConfigValue(
                    "embedding_field_prefix",
                    "The prefix of column names for embedding coordinates.",
                    "embed_",
                ),
                AutolockedConfigValue(
                    "source_color_field",
                    "The column name for plotted data point color.",
                    "__COLOR__",
                ),
                AutolockedConfigValue(
                    "source_alpha_field",
                    "The column name for plotted data point color alpha (opacity).",
                    "__ALPHA__",
                ),
                AutolockedConfigValue(
                    "search_score_field",
                    "The column name for data points' score from search widgets.",
                    "__SEARCH_SCORE__",
                ),
            ],
        ),
        Config(
            "data.values",
            [
                AutolockedConfigValue(
                    "abstain_decoded",
                    "The placeholder label indicating 'no label yet'.",
                    "ABSTAIN",
                ),
                AutolockedConfigValue(
                    "abstain_encoded",
                    "The encoded value of 'no label yet' which should almost always be -1, never 0 or positive.",
                    -1,
                ),
            ],
        ),
    ]
)
