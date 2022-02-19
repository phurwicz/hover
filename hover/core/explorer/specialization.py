"""
???+ note "Child classes which are `functionality`-by-`feature` products."
    This could resemble template specialization in C++.
"""
from .functionality import (
    BokehDataFinder,
    BokehDataAnnotator,
    BokehSoftLabelExplorer,
    BokehMarginExplorer,
    BokehSnorkelExplorer,
)
from .feature import BokehForText, BokehForAudio, BokehForImage
from bokeh.layouts import column, row


class BokehTextFinder(BokehDataFinder, BokehForText):
    """
    ???+ note "The text flavor of `BokehDataFinder`.""
    """

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataFinder.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(
                column(self.search_pos, self.search_neg),
                column(self.search_filter_box),
            ),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehTextAnnotator(BokehDataAnnotator, BokehForText):
    """
    ???+ note "The text flavor of `BokehDataAnnotator`.""
    """

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataAnnotator.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_pos, self.search_neg),
            row(self.annotator_input, self.annotator_apply),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehTextSoftLabel(BokehSoftLabelExplorer, BokehForText):
    """
    ???+ note "The text flavor of `BokehSoftLabelExplorer`.""
    """

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSoftLabelExplorer.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_pos, self.search_neg),
            row(self.score_filter),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehTextMargin(BokehMarginExplorer, BokehForText):
    """
    ???+ note "The text flavor of `BokehMarginExplorer`.""
    """

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehMarginExplorer.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_pos, self.search_neg),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehTextSnorkel(BokehSnorkelExplorer, BokehForText):
    """
    ???+ note "The text flavor of `BokehSnorkelExplorer`.""
    """

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSnorkelExplorer.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_pos, self.search_neg),
            row(self.lf_apply_trigger, self.lf_filter_trigger, self.lf_list_refresher),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehAudioFinder(BokehDataFinder, BokehForAudio):
    """
    ???+ note "The audio flavor of `BokehDataFinder`.""
    """

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataFinder.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(
                column(self.search_sim, self.search_threshold),
                column(self.search_filter_box),
            ),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehAudioAnnotator(BokehDataAnnotator, BokehForAudio):
    """
    ???+ note "The audio flavor of `BokehDataAnnotator`.""
    """

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataAnnotator.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_sim, self.search_threshold),
            row(self.annotator_input, self.annotator_apply),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehAudioSoftLabel(BokehSoftLabelExplorer, BokehForAudio):
    """
    ???+ note "The audio flavor of `BokehSoftLabelExplorer`.""
    """

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSoftLabelExplorer.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_sim, self.search_threshold),
            row(self.score_filter),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehAudioMargin(BokehMarginExplorer, BokehForAudio):
    """
    ???+ note "The audio flavor of `BokehMarginExplorer`.""
    """

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehMarginExplorer.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_sim, self.search_threshold),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehAudioSnorkel(BokehSnorkelExplorer, BokehForAudio):
    """
    ???+ note "The audio flavor of `BokehSnorkelExplorer`.""
    """

    TOOLTIP_KWARGS = BokehForAudio.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForAudio.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSnorkelExplorer.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_sim, self.search_threshold),
            row(self.lf_apply_trigger, self.lf_filter_trigger, self.lf_list_refresher),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehImageFinder(BokehDataFinder, BokehForImage):
    """
    ???+ note "The image flavor of `BokehDataFinder`.""
    """

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataFinder.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(
                column(self.search_sim, self.search_threshold),
                column(self.search_filter_box),
            ),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehImageAnnotator(BokehDataAnnotator, BokehForImage):
    """
    ???+ note "The image flavor of `BokehDataAnnotator`.""
    """

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataAnnotator.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_sim, self.search_threshold),
            row(self.annotator_input, self.annotator_apply),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehImageSoftLabel(BokehSoftLabelExplorer, BokehForImage):
    """
    ???+ note "The image flavor of `BokehSoftLabelExplorer`.""
    """

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSoftLabelExplorer.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_sim, self.search_threshold),
            row(self.score_filter),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehImageMargin(BokehMarginExplorer, BokehForImage):
    """
    ???+ note "The image flavor of `BokehMarginExplorer`.""
    """

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehMarginExplorer.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_sim, self.search_threshold),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)


class BokehImageSnorkel(BokehSnorkelExplorer, BokehForImage):
    """
    ???+ note "The image flavor of `BokehSnorkelExplorer`.""
    """

    TOOLTIP_KWARGS = BokehForImage.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForImage.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehSnorkelExplorer.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.subset_toggle_widget_column, self.selection_option_box),
            row(self.search_sim, self.search_threshold),
            row(self.lf_apply_trigger, self.lf_filter_trigger, self.lf_list_refresher),
            row(self.dropdown_x_axis, self.dropdown_y_axis),
            row(*self._dynamic_widgets.values()),
        )
        return column(*layout_rows)
