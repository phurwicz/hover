from bokeh.models import Div


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
<br>
Table:<br>
- a count of label populations, grouped by subset.<br>
"""


def dataset_help_widget():
    return Div(text=DATASET_HELP_HTML)
