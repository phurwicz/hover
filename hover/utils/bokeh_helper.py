"""
???+ note "Useful subroutines for working with bokeh in general."
"""
import warnings
from functools import wraps
from traceback import format_exc
from bokeh.models import PreText
from bokeh.layouts import column
from bokeh.palettes import Category10, Category20
from hover import module_config


def auto_label_color(labels):
    """
    ???+ note "Create a label->hex color mapping dict."
    """
    use_labels = set(labels)
    use_labels.discard(module_config.ABSTAIN_DECODED)
    use_labels = sorted(use_labels, reverse=False)

    assert len(use_labels) <= 20, "Too many labels to support (max at 20)"
    palette = Category10[10] if len(use_labels) <= 10 else Category20[20]
    color_dict = {
        module_config.ABSTAIN_DECODED: "#dcdcdc",  # gainsboro hex code
        **{_l: _c for _l, _c in zip(use_labels, palette)},
    }
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
                    spinner.text += "."

                def load():
                    try:
                        bokeh_model = func(*args, **kwargs)
                        # remove spinner and its update
                        try:
                            doc.remove_periodic_callback(progress)
                        except Exception as e:
                            warnings.warn(
                                f"@servable: trying to remove periodic callback, got {type(e)}: {e}"
                            )
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
