"""
Module root where constants get configured.
"""
import re
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
                    validation=lambda x: isinstance(x, str),
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
                    validation=lambda x: bool(re.match(r"^\#[0-9a-fA-F]{6}$", x)),
                ),
                AutolockedConfigValue(
                    "bokeh_palette",
                    "The bokeh color palette to use for plotting.",
                    Turbo256,
                    validation=lambda x: hasattr(x, "__iter__"),
                ),
                AutolockedConfigValue(
                    "table_img_style",
                    "HTML style of images shown in selection tables.",
                    "max-height: 100%; max-width: 100%; object-fit: contain",
                    preprocessor=lambda x: re.sub(r"(^[\'\"]|[\'\"]$)", "", x),
                ),
                AutolockedConfigValue(
                    "tooltip_img_style",
                    "HTML style of images shown in mouse-over-data-point tooltips.",
                    "float: left; margin: 2px 2px 2px 2px; width: 60px; height: 60px;",
                    preprocessor=lambda x: re.sub(r"(^[\'\"]|[\'\"]$)", "", x),
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
                    validation=lambda x: x in ["umap", "ivis"],
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
                    validation=lambda x: isinstance(x, str),
                ),
                AutolockedConfigValue(
                    "dataset_subset_field",
                    "The column name for dataset subsets.",
                    "SUBSET",
                    validation=lambda x: isinstance(x, str),
                ),
                AutolockedConfigValue(
                    "embedding_field_prefix",
                    "The prefix of column names for embedding coordinates.",
                    "embed_",
                    validation=lambda x: isinstance(x, str),
                ),
                AutolockedConfigValue(
                    "source_color_field",
                    "The column name for plotted data point color.",
                    "__COLOR__",
                    validation=lambda x: isinstance(x, str),
                ),
                AutolockedConfigValue(
                    "source_alpha_field",
                    "The column name for plotted data point color alpha (opacity).",
                    "__ALPHA__",
                    validation=lambda x: isinstance(x, str),
                ),
                AutolockedConfigValue(
                    "search_score_field",
                    "The column name for data points' score from search widgets.",
                    "__SEARCH_SCORE__",
                    validation=lambda x: isinstance(x, str),
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
                    validation=lambda x: isinstance(x, str),
                ),
                AutolockedConfigValue(
                    "abstain_encoded",
                    "The encoded value of 'no label yet' which should almost always be -1, never 0 or positive.",
                    -1,
                    validation=lambda x: isinstance(x, int) and x < 0,
                ),
            ],
        ),
    ]
)
