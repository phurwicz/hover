from bokeh.models import Div


DATASET_SUBSET_FIELD = "SUBSET"

COLOR_GLYPH_TEMPLATE = """
<p style="color:<%= value %>;">
    <%= "&#9608;" %>
</p>
"""


def dataset_help_widget():
    text = 'Dataset Widgets <a href="https://phurwicz.github.io/hover/">Help</a>'
    return Div(text=text)
