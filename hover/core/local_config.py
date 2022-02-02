from bokeh.models import (
    Div,
    TableColumn,
    CellEditor,
    HTMLTemplateFormatter,
)


DATASET_SUBSET_FIELD = "SUBSET"

COLOR_GLYPH_TEMPLATE = """
<p style="color:<%= value %>;">
    <%= "&#9608;&#9608;&#9608;" %>
</p>
"""


def dataset_help_widget():
    text = 'Dataset Widgets <a href="https://phurwicz.github.io/hover/" target="_blank" rel="noreferrer noopener">Help</a>'
    return Div(text=text)


def dataset_default_sel_table_columns(feature_key):
    """
    ???+ note "Default `SupervisableDataset` selection table columns based on feature type."

        Always allow multi-selection and editing. Based on feature type:
        -   increases row height for viewing images.
    """
    # disable editing the feature through a blank editor
    feature_col_kwargs = dict(editor=CellEditor())
    if feature_key == "text":
        pass
    elif feature_key == "image":
        feature_col_kwargs["width"] = 200
        feature_col_kwargs["formatter"] = HTMLTemplateFormatter(
            template="""<img src=<%= value %>>""",
        )
    elif feature_key == "audio":
        feature_col_kwargs["width"] = 50
        feature_col_kwargs["formatter"] = HTMLTemplateFormatter(
            template="""<audio controls preload="auto" src=<%= value %>></audio>""",
        )
    else:
        raise ValueError(f"Unsupported feature type {feature_key}")

    columns = [
        TableColumn(field=feature_key, title=feature_key, **feature_col_kwargs),
        TableColumn(field="label", title="label"),
    ]
    return columns


def dataset_default_sel_table_kwargs(feature_key):
    """
    ???+ note "Default `SupervisableDataset` selection table kwargs based on feature type."

        Always allow multi-selection and editing. Based on feature type:
        -   increases row height for viewing images.
    """
    kwargs = dict(selectable="checkbox", editable=True)
    if feature_key == "text":
        pass
    elif feature_key == "image":
        kwargs["row_height"] = 200
    elif feature_key == "audio":
        pass
    else:
        raise ValueError(f"Unsupported feature type {feature_key}")

    return kwargs
