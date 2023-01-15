"""
Module root where constants get configured.
"""
import os
from flexmod import AutolockedConfigValue, AutolockedConfig, AutolockedConfigIndex

config = AutolockedConfigIndex(
    [
        AutolockedConfig(
            "io",
            [
                AutolockedConfigValue(
                    "data_save_dir",
                    "The directory path for saving labeled data.",
                    ".",
                ),
            ],
        ),
        AutolockedConfig(
            "visual",
            [
                AutolockedConfigValue(
                    "abstain_hexcolor",
                    "Hex code of RGB color.",
                    "#dcdcdc",
                ),
                AutolockedConfigValue(
                    "bokeh_palette_name",
                    "The bokeh color palette to use for plotting.",
                    "Turbo256",
                ),
            ],
        ),
        AutolockedConfig(
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
                    "The column name for dataset subsets.",
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
        AutolockedConfig(
            "data.values",
            [
                AutolockedConfigValue(
                    "abstain_decoded",
                    "The placeholder label indicating 'no label yet'.",
                    "#dcdcdc",
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
