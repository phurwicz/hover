import hover
from hover.config_constants import (
    ConfigSection as Section,
    ConfigKey as Key,
)

SOURCE_COLOR_FIELD = hover.config[Section.DATA_COLUMNS][Key.SOURCE_COLOR_FIELD]
SOURCE_ALPHA_FIELD = hover.config[Section.DATA_COLUMNS][Key.SOURCE_ALPHA_FIELD]
SEARCH_SCORE_FIELD = hover.config[Section.DATA_COLUMNS][Key.SEARCH_SCORE_FIELD]

TOOLTIP_IMG_STYLE = hover.config[Section.VISUAL][Key.TOOLTIP_IMG_STYLE]

SEARCH_MATCH_HEXCOLOR = hover.config[Section.VISUAL][Key.SEARCH_MATCH_HEXCOLOR]
DATAPOINT_BASE_SIZE = hover.config[Section.VISUAL][Key.DATAPOINT_BASE_SIZE]
SEARCH_DATAPOINT_SIZE_PARAMS = (
    "size",
    DATAPOINT_BASE_SIZE + 3,
    DATAPOINT_BASE_SIZE - 2,
    DATAPOINT_BASE_SIZE,
)
