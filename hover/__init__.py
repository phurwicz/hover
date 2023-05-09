"""
Module root where constants get configured.
"""
import operator
from .config_constants import (
    ConfigSection,
    ConfigKey,
    Validator,
    Preprocessor,
)
from flexmod import AutolockedConfigValue, Config, ConfigIndex
from bokeh.palettes import Turbo256


config = ConfigIndex(
    [
        Config(
            ConfigSection.IO,
            [
                AutolockedConfigValue(
                    ConfigKey.DATA_SAVE_DIR,
                    "The directory path for saving labeled data.",
                    ".",
                    validation=Validator.is_str,
                ),
            ],
        ),
        Config(
            ConfigSection.BACKEND,
            [
                AutolockedConfigValue(
                    ConfigKey.DATAFRAME_LIBRARY,
                    "The library to use for internal dataframes. Must be 'pandas' or 'polars'.",
                    "pandas",
                    validation=Validator.is_supported_dataframe_library,
                    preprocessor=Preprocessor.lower,
                ),
            ],
        ),
        Config(
            ConfigSection.VISUAL,
            [
                AutolockedConfigValue(
                    ConfigKey.ABSTAIN_HEXCOLOR,
                    "Hex code of RGB color.",
                    "#dcdcdc",
                    validation=Validator.is_hex_color,
                    preprocessor=Preprocessor.lower,
                ),
                AutolockedConfigValue(
                    ConfigKey.BOKEH_PALETTE,
                    "The bokeh color palette to use for plotting. This should be a list of hex color codes.",
                    Turbo256,
                    validation=Validator.is_iterable_of_hex_color,
                ),
                AutolockedConfigValue(
                    ConfigKey.BOKEH_PALETTE_USAGE,
                    "Specify how colors from the palette should be chosen when there are fewer categories than colors. This needs to be 'iterate' or 'linspace'",
                    "linspace",
                    validation=Validator.is_supported_traversal_mode,
                    preprocessor=Preprocessor.lower,
                ),
                AutolockedConfigValue(
                    ConfigKey.TABLE_IMG_STYLE,
                    "HTML style of images shown in selection tables.",
                    "max-height: 100%; max-width: 100%; object-fit: contain",
                    preprocessor=Preprocessor.remove_quote_at_ends,
                ),
                AutolockedConfigValue(
                    ConfigKey.TOOLTIP_IMG_STYLE,
                    "HTML style of images shown in mouse-over-data-point tooltips.",
                    "float: left; margin: 2px 2px 2px 2px; width: 60px; height: 60px;",
                    preprocessor=Preprocessor.remove_quote_at_ends,
                ),
                AutolockedConfigValue(
                    ConfigKey.SEARCH_MATCH_HEXCOLOR,
                    "Hex code of the color to use on the data points matching a search widget. Note that only the `finder` plot changes color.",
                    "#ee4443",
                    validation=Validator.is_hex_color,
                    preprocessor=Preprocessor.lower,
                ),
                AutolockedConfigValue(
                    ConfigKey.DATAPOINT_BASE_SIZE,
                    "The base (bokeh) size of data points in the scatter plot. Size may change to highlight/dehighlight data points. Must be at least 3.",
                    7,
                    validation=Validator.is_int_and_compare(operator.ge, 3),
                ),
            ],
        ),
        Config(
            ConfigSection.DATA_EMBEDDING,
            [
                AutolockedConfigValue(
                    ConfigKey.DEFAULT_REDUCTION_METHOD,
                    "Default method for dimensionality reduction. Currently either 'umap' or 'ivis'.",
                    "umap",
                    validation=Validator.is_supported_dimensionality_reduction,
                    preprocessor=Preprocessor.lower,
                ),
            ],
        ),
        Config(
            ConfigSection.DATA_COLUMNS,
            [
                AutolockedConfigValue(
                    ConfigKey.ENCODED_LABEL_KEY,
                    "The column name for the encoded label.",
                    "label_encoded",
                    validation=Validator.is_str,
                ),
                AutolockedConfigValue(
                    ConfigKey.DATASET_SUBSET_FIELD,
                    "The column name for dataset subsets.",
                    "SUBSET",
                    validation=Validator.is_str,
                ),
                AutolockedConfigValue(
                    ConfigKey.EMBEDDING_FIELD_PREFIX,
                    "The prefix of column names for embedding coordinates.",
                    "embed_",
                    validation=Validator.is_str,
                ),
                AutolockedConfigValue(
                    ConfigKey.SOURCE_COLOR_FIELD,
                    "The column name for plotted data point color.",
                    "__COLOR__",
                    validation=Validator.is_str,
                ),
                AutolockedConfigValue(
                    ConfigKey.SOURCE_ALPHA_FIELD,
                    "The column name for plotted data point color alpha (opacity).",
                    "__ALPHA__",
                    validation=Validator.is_str,
                ),
                AutolockedConfigValue(
                    ConfigKey.SEARCH_SCORE_FIELD,
                    "The column name for data points' score from search widgets.",
                    "__SEARCH_SCORE__",
                    validation=Validator.is_str,
                ),
            ],
        ),
        Config(
            ConfigSection.DATA_VALUES,
            [
                AutolockedConfigValue(
                    ConfigKey.ABSTAIN_DECODED,
                    "The placeholder label indicating 'no label yet'.",
                    "ABSTAIN",
                    validation=Validator.is_str,
                ),
                AutolockedConfigValue(
                    ConfigKey.ABSTAIN_ENCODED,
                    "The encoded value of 'no label yet' which should almost always be -1, never 0 or positive.",
                    -1,
                    validation=Validator.is_int_and_compare(operator.lt, 0),
                ),
            ],
        ),
    ]
)
