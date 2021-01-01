"""Interactive explorers mostly based on Bokeh."""
from .functionality import (
    BokehDataFinder,
    BokehDataAnnotator,
    BokehSoftLabelExplorer,
    BokehMarginExplorer,
    BokehSnorkelExplorer,
)
from .feature import BokehForText, BokehForAudio, BokehForImage
from deprecated import deprecated


class BokehTextFinder(BokehDataFinder, BokehForText):
    """The text flavor of BokehDataFinder."""

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataFinder.SUBSET_GLYPH_KWARGS


class BokehTextAnnotator(BokehDataAnnotator, BokehForText):
    """The text flavor of BokehDataAnnotator."""

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataAnnotator.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        from bokeh.layouts import column, row

        layout_rows = (
            row(self.search_pos, self.search_neg),
            row(self.data_key_button_group),
            row(self.annotator_input, self.annotator_apply, self.annotator_export),
        )
        return column(*layout_rows)


class BokehTextSoftLabel(BokehSoftLabelExplorer, BokehForText):
    """The text flavor of BokehSoftLabelExplorer."""

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSoftLabelExplorer.SUBSET_GLYPH_KWARGS


class BokehTextMargin(BokehMarginExplorer, BokehForText):
    """The text flavor of BokehMarginExplorer."""

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehMarginExplorer.SUBSET_GLYPH_KWARGS


class BokehTextSnorkel(BokehSnorkelExplorer, BokehForText):
    """The text flavor of BokehSnorkelExplorer."""

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSnorkelExplorer.SUBSET_GLYPH_KWARGS


class BokehAudioFinder(BokehDataFinder, BokehForAudio):
    """The text flavor of BokehDataFinder."""

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataFinder.SUBSET_GLYPH_KWARGS


class BokehAudioAnnotator(BokehDataAnnotator, BokehForAudio):
    """The text flavor of BokehDataAnnotator."""

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataAnnotator.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        from bokeh.layouts import column, row

        layout_rows = (
            row(self.search_pos, self.search_neg),
            row(self.data_key_button_group),
            row(self.annotator_input, self.annotator_apply, self.annotator_export),
        )
        return column(*layout_rows)


class BokehAudioSoftLabel(BokehSoftLabelExplorer, BokehForAudio):
    """The text flavor of BokehSoftLabelExplorer."""

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSoftLabelExplorer.SUBSET_GLYPH_KWARGS


class BokehAudioMargin(BokehMarginExplorer, BokehForAudio):
    """The text flavor of BokehMarginExplorer."""

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehMarginExplorer.SUBSET_GLYPH_KWARGS


class BokehAudioSnorkel(BokehSnorkelExplorer, BokehForAudio):
    """The text flavor of BokehSnorkelExplorer."""

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSnorkelExplorer.SUBSET_GLYPH_KWARGS


class BokehImageFinder(BokehDataFinder, BokehForImage):
    """The text flavor of BokehDataFinder."""

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataFinder.SUBSET_GLYPH_KWARGS


class BokehImageAnnotator(BokehDataAnnotator, BokehForImage):
    """The text flavor of BokehDataAnnotator."""

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataAnnotator.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        from bokeh.layouts import column, row

        layout_rows = (
            row(self.search_pos, self.search_neg),
            row(self.data_key_button_group),
            row(self.annotator_input, self.annotator_apply, self.annotator_export),
        )
        return column(*layout_rows)


class BokehImageSoftLabel(BokehSoftLabelExplorer, BokehForImage):
    """The text flavor of BokehSoftLabelExplorer."""

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSoftLabelExplorer.SUBSET_GLYPH_KWARGS


class BokehImageMargin(BokehMarginExplorer, BokehForImage):
    """The text flavor of BokehMarginExplorer."""

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehMarginExplorer.SUBSET_GLYPH_KWARGS


class BokehImageSnorkel(BokehSnorkelExplorer, BokehForImage):
    """The text flavor of BokehSnorkelExplorer."""

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSnorkelExplorer.SUBSET_GLYPH_KWARGS


@deprecated(
    version="0.4.0",
    reason="will be removed in a future version; please use BokehTextFinder instead.",
)
class BokehCorpusExplorer(BokehTextFinder):
    pass


@deprecated(
    version="0.4.0",
    reason="will be removed in a future version; please use BokehTextFinder instead.",
)
class BokehCorpusAnnotator(BokehTextAnnotator):
    pass
