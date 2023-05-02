import hover
from hover.config_constants import (
    ConfigSection as Section,
    ConfigKey as Key,
)

SOURCE_COLOR_FIELD = hover.config[Section.DATA_COLUMNS][Key.SOURCE_COLOR_FIELD]
SOURCE_ALPHA_FIELD = hover.config[Section.DATA_COLUMNS][Key.SOURCE_ALPHA_FIELD]
SEARCH_SCORE_FIELD = hover.config[Section.DATA_COLUMNS][Key.SEARCH_SCORE_FIELD]

TOOLTIP_IMG_STYLE = hover.config[Section.VISUAL][Key.TOOLTIP_IMG_STYLE]
