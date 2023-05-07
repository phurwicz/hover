import re


class ConfigSection:
    IO = "io"
    BACKEND = "backend"
    VISUAL = "visual"
    DATA_EMBEDDING = "data.embedding"
    DATA_COLUMNS = "data.columns"
    DATA_VALUES = "data.values"


class ConfigKey:
    DATA_SAVE_DIR = "data_save_dir"
    DATAFRAME_LIBRARY = "dataframe_library"
    ABSTAIN_HEXCOLOR = "abstain_hexcolor"
    BOKEH_PALETTE = "bokeh_palette"
    BOKEH_PALETTE_USAGE = "bokeh_palette_usage"
    TABLE_IMG_STYLE = "table_img_style"
    TOOLTIP_IMG_STYLE = "tooltip_img_style"
    SEARCH_MATCH_HEXCOLOR = "search_match_hexcolor"
    DATAPOINT_BASE_SIZE = "datapoint_base_size"
    DEFAULT_REDUCTION_METHOD = "default_reduction_method"
    ENCODED_LABEL_KEY = "encoded_label_key"
    DATASET_SUBSET_FIELD = "dataset_subset_field"
    EMBEDDING_FIELD_PREFIX = "embedding_field_prefix"
    SOURCE_COLOR_FIELD = "source_color_field"
    SOURCE_ALPHA_FIELD = "source_alpha_field"
    SEARCH_SCORE_FIELD = "search_score_field"
    ABSTAIN_DECODED = "abstain_decoded"
    ABSTAIN_ENCODED = "abstain_encoded"


class Validator:
    @staticmethod
    def is_hex_color(x):
        return bool(re.match(r"^\#[0-9a-fA-F]{6}$", x))

    @staticmethod
    def is_iterable(x):
        return hasattr(x, "__iter__")

    @staticmethod
    def is_iterable_of_hex_color(x):
        if not Validator.is_iterable(x):
            return False
        for i in x:
            if not Validator.is_hex_color(i):
                return False
        return True

    @staticmethod
    def is_supported_dataframe_library(x):
        return x in ["pandas", "polars"]

    @staticmethod
    def is_supported_dimensionality_reduction(x):
        return x in ["umap", "ivis"]

    @staticmethod
    def is_supported_traversal_mode(x):
        return x in ["iterate", "linspace"]

    @staticmethod
    def is_str(x):
        return isinstance(x, str)

    @staticmethod
    def is_int_and_compare(op, value):
        def func(x):
            return isinstance(x, int) and op(x, value)

        return func


class Preprocessor:
    @staticmethod
    def remove_quote_at_ends(x):
        return re.sub(r"(^[\'\"]|[\'\"]$)", "", x)

    @staticmethod
    def lower(x):
        return x.lower()
