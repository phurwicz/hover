import hover
from .config_constants import (
    ConfigSection as Section,
    ConfigKey as Key,
)
from .utils.dataframe import (
    PandasDataframe,
    PolarsDataframe,
)

# dataframe implementation
DataFrame = (
    PandasDataframe
    if hover.config[Section.BACKEND][Key.DATAFRAME_LIBRARY].lower() == "pandas"
    else PolarsDataframe
)

# constants for the abstain mechanism
ABSTAIN_DECODED = hover.config[Section.DATA_VALUES][Key.ABSTAIN_DECODED]
ABSTAIN_ENCODED = hover.config[Section.DATA_VALUES][Key.ABSTAIN_ENCODED]
ABSTAIN_HEXCOLOR = hover.config[Section.VISUAL][Key.ABSTAIN_HEXCOLOR]

# constants for label encoding mechanism
ENCODED_LABEL_KEY = hover.config[Section.DATA_COLUMNS][Key.ENCODED_LABEL_KEY]

# constants for saving work
DATA_SAVE_DIR = hover.config[Section.IO][Key.DATA_SAVE_DIR]
