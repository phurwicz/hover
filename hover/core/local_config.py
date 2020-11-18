# bokeh hovertool template
def bokeh_hover_tooltip(
    label=False, text=False, image=False, coords=True, index=True, custom={}
):
    """
    Create a Bokeh hover tooltip from a template.
    """
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
        <div style="word-wrap: break-word; width: 800px; text-overflow: ellipsis; line-height: 80%">
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
        assert _field.startswith("@")
        tooltip += f"""
        <div>
            <span style="font-size: 12px; color: #606;">
                {_key}: {_field}
            </span>
        </div>
        """
    tooltip += suffix
    return tooltip
