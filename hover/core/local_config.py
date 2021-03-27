from bokeh.models import Div


DATASET_SUBSET_FIELD = "SUBSET"

COLOR_GLYPH_TEMPLATE = """
<p style="color:<%= value %>;">
    <%= "&#9608;" %>
</p>
"""

DATASET_HELP_HTML = """Hover - dataset visual elements<br>
<br>
Buttons:<br>
- Push: "dataset -> plots" push of data points;<br>
- Commit: "selected in plot -> selected subset" commit of data points;<br>
- Dedup: "deduplication, keeping test -> dev -> train -> raw".<br>
- View Selected: "view data entries selected in the explorer".<br>
<br>
Tables:<br>
- a count of label populations, grouped by subset.<br>
- a view of selected data points.<br>
"""


def dataset_help_widget():
    return Div(text=DATASET_HELP_HTML)
