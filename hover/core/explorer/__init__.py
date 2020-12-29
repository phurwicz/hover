"""Interactive explorers mostly based on Bokeh."""
from .functionality import (
    BokehDataFinder,
    BokehDataAnnotator,
    BokehSoftLabelExplorer,
    BokehMarginExplorer,
    BokehSnorkelExplorer,
)
from .medium import BokehForCorpus
from deprecated import deprecated


class BokehCorpusFinder(BokehDataFinder, BokehForCorpus):
    """The text flavor of BokehDataFinder."""

    TOOLTIP_KWARGS = BokehForCorpus.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForCorpus.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataFinder.SUBSET_GLYPH_KWARGS


class BokehCorpusAnnotator(BokehDataAnnotator, BokehForCorpus):
    """The text flavor of BokehDataAnnotator."""

    TOOLTIP_KWARGS = BokehForCorpus.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForCorpus.MANDATORY_COLUMNS
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


class BokehCorpusSoftLabel(BokehSoftLabelExplorer, BokehForCorpus):
    """The text flavor of BokehSoftLabelExplorer."""

    TOOLTIP_KWARGS = BokehForCorpus.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForCorpus.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSoftLabelExplorer.SUBSET_GLYPH_KWARGS


class BokehCorpusMargin(BokehMarginExplorer, BokehForCorpus):
    """The text flavor of BokehMarginExplorer."""

    TOOLTIP_KWARGS = BokehForCorpus.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForCorpus.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehMarginExplorer.SUBSET_GLYPH_KWARGS


class BokehCorpusSnorkel(BokehSnorkelExplorer, BokehForCorpus):
    """The text flavor of BokehSnorkelExplorer."""

    TOOLTIP_KWARGS = BokehForCorpus.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForCorpus.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSnorkelExplorer.SUBSET_GLYPH_KWARGS


@deprecated(
    version="0.4.0",
    reason="will be removed in a future version; please use BokehCorpusFinder instead.",
)
class BokehCorpusExplorer(BokehCorpusFinder):
    pass
