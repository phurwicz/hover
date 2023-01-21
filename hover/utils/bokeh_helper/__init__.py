"""
???+ note "Useful subroutines for working with bokeh in general."
"""
import os
import hover
import numpy as np
from functools import wraps
from traceback import format_exc
from urllib.parse import urljoin, urlparse
from bokeh.models import PreText
from bokeh.layouts import column
from hover import module_config
from .local_config import (
    TOOLTIP_TEXT_TEMPLATE,
    TOOLTIP_IMAGE_TEMPLATE,
    TOOLTIP_AUDIO_TEMPLATE,
    TOOLTIP_CUSTOM_TEMPLATE,
    TOOLTIP_LABEL_TEMPLATE,
    TOOLTIP_COORDS_DIV,
    TOOLTIP_INDEX_DIV,
)


def auto_label_color(labels):
    """
    ???+ note "Create a label->hex color mapping dict."
    """
    use_labels = set(labels)
    use_labels.discard(module_config.ABSTAIN_DECODED)
    use_labels = sorted(use_labels, reverse=False)

    palette = hover.config["visual"]["bokeh_palette"]
    assert len(use_labels) <= len(
        palette
    ), f"Too many labels to support (max at {len(palette)})"

    use_palette_idx = np.linspace(0.0, len(palette), len(use_labels) + 2).astype(int)[
        1:-1
    ]
    assert len(set(use_palette_idx)) == len(
        use_palette_idx
    ), "Found repeated palette index"
    assert len(use_palette_idx) == len(
        use_labels
    ), "Number of labels vs. palette colors must equal."

    use_palette = [palette[i] for i in use_palette_idx]
    color_dict = {
        module_config.ABSTAIN_DECODED: module_config.ABSTAIN_HEXCOLOR,
        **{_l: _c for _l, _c in zip(use_labels, use_palette)},
    }
    print(color_dict)
    return color_dict


def servable(title=None):
    """
    ???+ note "Create a decorator which returns an app (or "handle" function) to be passed to bokeh."

        Usage:

        First wrap a function that creates bokeh plot elements:

        ```python
        @servable()
        def dummy(*args, **kwargs):
            from hover.core.explorer import BokehCorpusAnnotator
            annotator = BokehCorpusAnnotator(*args, **kwargs)
            annotator.plot()

            return annotator.view()
        ```

        Then serve the app in your preferred setting:

        === "inline"
            ```python
            # in a Jupyter cell

            from bokeh.io import show, output_notebook
            output_notebook()
            show(dummy(*args, **kwargs))
            ```

        === "bokeh serve"
            ```python
            # in <your-bokeh-app-dir>/main.py

            from bokeh.io import curdoc
            doc = curdoc()
            dummy(*args, **kwargs)(doc)
            ```

        === "embedded app"
            ```python
            # anywhere in your use case

            from bokeh.server.server import Server
            app_dict = {
                'my-app': dummy(*args, **kwargs),
                'my-other-app': dummy(*args, **kwargs),
            }
            server = Server(app_dict)
            server.start()
            ```
    """

    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            def handle(doc):
                """
                Note that the handle must create a brand new bokeh model every time it is called.

                Reference: https://github.com/bokeh/bokeh/issues/8579
                """
                spinner = PreText(text="loading...")
                layout = column(spinner)

                def progress():
                    """
                    If still loading, show some progress.
                    """
                    if spinner in layout.children:
                        spinner.text += "."

                def load():
                    try:
                        bokeh_model = func(*args, **kwargs)
                        layout.children.append(bokeh_model)
                        layout.children.pop(0)
                    except Exception as e:
                        # exception handling
                        message = PreText(text=f"{type(e)}: {e}\n{format_exc()}")
                        layout.children.append(message)

                doc.add_root(layout)
                doc.add_periodic_callback(progress, 5000)
                doc.add_timeout_callback(load, 500)
                doc.title = title or func.__name__

            return handle

        return wrapped

    return wrapper


def show_as_interactive(obj, **kwargs):
    """
    ???+ note "Wrap a bokeh LayoutDOM as an application to allow Python callbacks."

        Must have the same signature as `bokeh.io.show()`[https://docs.bokeh.org/en/latest/docs/reference/io.html#bokeh.io.show].
    """
    from bokeh.io import show
    from bokeh.models.layouts import LayoutDOM

    assert isinstance(obj, LayoutDOM), f"Expected Bokeh LayoutDOM, got {type(obj)}"

    def handle(doc):
        doc.add_root(column(obj))

    return show(handle, **kwargs)


def bokeh_hover_tooltip(
    label=None,
    text=None,
    image=None,
    audio=None,
    coords=True,
    index=True,
    custom=None,
):
    """
    ???+ note "Create a Bokeh hover tooltip from a template."
    """
    # initialize default values of mutable type
    label = label or dict(Label=label)
    text = text or dict()
    image = image or dict()
    audio = audio or dict()
    custom = custom or dict()

    # prepare encapsulation of a div box and an associated script
    divbox_prefix = """<div class="out tooltip">\n"""
    divbox_suffix = """</div>\n"""
    script_prefix = """<script>\n"""
    script_suffix = """</script>\n"""

    # dynamically add contents to the div box and the script
    divbox = divbox_prefix
    script = script_prefix

    for _field, _key in label.items():
        divbox += TOOLTIP_LABEL_TEMPLATE.format(field=_field, key=_key)

    for _field, _key in text.items():
        divbox += TOOLTIP_TEXT_TEMPLATE.format(field=_field, key=_key)

    for _field, _style in image.items():
        divbox += TOOLTIP_IMAGE_TEMPLATE.format(field=_field, style=_style)

    for _field, _option in audio.items():
        divbox += TOOLTIP_AUDIO_TEMPLATE.format(field=_field, option=_option)

    if coords:
        divbox += TOOLTIP_COORDS_DIV

    if index:
        divbox += TOOLTIP_INDEX_DIV

    for _field, _key in custom.items():
        divbox += TOOLTIP_CUSTOM_TEMPLATE.format(field=_field, key=_key)

    divbox += divbox_suffix
    script += script_suffix
    return divbox + script


def binder_proxy_app_url(app_path, port=5006):
    """
    ???+ note "Find the URL of Bokeh server app in the current Binder session."

        Intended for visiting a Binder-hosted Bokeh server app.

        Will NOT work outside of Binder.
    """

    service_url_path = os.environ.get(
        "JUPYTERHUB_SERVICE_PREFIX", "/user/hover-binder/"
    )
    proxy_url_path = f"proxy/{port}/{app_path}"

    base_url = "https://hub.gke2.mybinder.org"
    user_url_path = urljoin(service_url_path, proxy_url_path)
    full_url = urljoin(base_url, user_url_path)
    return full_url


def remote_jupyter_proxy_url(port):
    """
    ???+ note "Callable to configure Bokeh's show method when using a proxy (JupyterHub)."

        Intended for rendering a in-notebook Bokeh app.

        Usage:

        ```python
        # show(plot)
        show(plot, notebook_url=remote_jupyter_proxy_url)
        ```
    """

    # find JupyterHub base (external) url, default to Binder
    base_url = os.environ.get("JUPYTERHUB_BASE_URL", "https://hub.gke2.mybinder.org")
    host = urlparse(base_url).netloc

    if port is None:
        return host

    service_url_path = os.environ.get(
        "JUPYTERHUB_SERVICE_PREFIX", "/user/hover-binder/"
    )
    proxy_url_path = f"proxy/{port}"

    user_url = urljoin(base_url, service_url_path)
    full_url = urljoin(user_url, proxy_url_path)
    return full_url
