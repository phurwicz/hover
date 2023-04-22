import re


class ConfigSection:
    IO = "io"
    VISUAL = "visual"
    DATA_EMBEDDING = "data.embedding"
    DATA_COLUMNS = "data.columns"
    DATA_VALUES = "data.values"


class ConfigKey:
    DATA_SAVE_DIR = "data_save_dir"
    ABSTAIN_HEXCOLOR = "abstain_hexcolor"
    BOKEH_PALETTE = "bokeh_palette"
    BOKEH_PALETTE_USAGE = "bokeh_palette_usage"
    TABLE_IMG_STYLE = "table_img_style"
    TOOLTIP_IMG_STYLE = "tooltip_img_style"
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
    def is_supported_dimensionality_reduction(x):
        return x.lower() in ["umap", "ivis"]

    @staticmethod
    def is_supported_traversal_mode(x):
        return x.lower() in ["iterate", "linspace"]

    @staticmethod
    def is_str(x):
        return isinstance(x, str)

    @staticmethod
    def is_negative_int(x):
        return isinstance(x, int) and x < 0


class Preprocessor:
    @staticmethod
    def remove_quote_at_ends(x):
        return re.sub(r"(^[\'\"]|[\'\"]$)", "", x)
