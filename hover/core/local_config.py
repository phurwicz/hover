from bokeh.models import Div


def bokeh_hover_tooltip(
    label=False, text=False, image=False, coords=True, index=True, custom=None
):
    """
    Create a Bokeh hover tooltip from a template.

    :param label: whether to expect and show a "label" field.
    :param text: whether to expect and show a "text" field.
    :param image: whether to expect an "image" (path) field and show the corresponding images.
    :param coords: whether to show xy-coordinates.
    :param index: whether to show indices in the dataset.
    :param custom: {display: column} mapping of additional (text) tooltips.
    """
    custom = custom or dict()
    prefix = """<div>\n"""
    suffix = """</div>\n"""

    tooltip = prefix
    if label:
        tooltip += """
        <div>
            <span style="font-size: 16px; color: #966;">
                Label: @label
            </span>
        </div>
        """
    if text:
        tooltip += """
        <div style="word-wrap: break-word; width: 95%; text-overflow: ellipsis; line-height: 90%">
            <span style="font-size: 11px;">
                Text: @text
            </span>
        </div>
        """
    if image:
        tooltip += """
        <div>
            <img
                src="@imgs" height="60" alt="@imgs" width="60"
                style="float: left; margin: 0px 5px 5px 0px;"
                border="2"
            ></img>
        </div>
        """
    if coords:
        tooltip += """
        <div>
            <span style="font-size: 12px; color: #060;">
                Coordinates: ($x, $y)
            </span>
        </div>
        """
    if index:
        tooltip += """
        <div>
            <span style="font-size: 12px; color: #066;">
                Index: [$index]
            </span>
        </div>
        """
    for _key, _field in custom.items():
        tooltip += f"""
        <div>
            <span style="font-size: 12px; color: #606;">
                {_key}: @{_field}
            </span>
        </div>
        """
    tooltip += suffix
    return tooltip


dataset_help_html = """Hover - dataset visual elements<br>
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
    return Div(text=dataset_help_html)
