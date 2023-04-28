import hover
from hover.config_constants import (
    ConfigSection as Section,
    ConfigKey as Key,
)


BOKEH_PALETTE = hover.config[Section.VISUAL][Key.BOKEH_PALETTE]
BOKEH_PALETTE_USAGE = hover.config[Section.VISUAL][Key.BOKEH_PALETTE_USAGE]

TOOLTIP_TEXT_TEMPLATE = """
    <div style="word-wrap: break-word; width: 95%; text-overflow: ellipsis; line-height: 90%">
        <span style="font-size: 11px;">
            {key}: @{field}
        </span>
    </div>
"""

TOOLTIP_IMAGE_TEMPLATE = """
    <div>
        <img
            src="@{field}" alt="@{field}" style="{style}"
            border="1"
        ></img>
    </div>
"""

TOOLTIP_AUDIO_TEMPLATE = """
    <div>
        <audio
            autoplay
            preload="auto"
            {option}
            src="@{field}"
        ></audio>
    </div>
"""

TOOLTIP_CUSTOM_TEMPLATE = """
    <div>
        <span style="font-size: 12px; color: #606;">
            {key}: @{field}
        </span>
    </div>
"""

TOOLTIP_LABEL_TEMPLATE = """
    <div>
        <span style="font-size: 16px; color: #966;">
            {key}: @{field}
        </span>
    </div>
"""

TOOLTIP_COORDS_DIV = """
    <div>
        <span style="font-size: 12px; color: #060;">
            Coordinates: ($x, $y)
        </span>
    </div>
"""

TOOLTIP_INDEX_DIV = """
    <div>
        <span style="font-size: 12px; color: #066;">
            Index: [$index]
        </span>
    </div>
"""
