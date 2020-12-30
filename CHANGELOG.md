0.4.0 - Upcoming
==================

### Features

-   Added loading text before each `hover.recipes.<recipe>` renders in BokehJS.
    -   displays an additional dot every a few (5) seconds.
    -   this can be useful for visiting a remote Bokeh server.
-   **Partial** towards image data support
    -   `hover.core.explorer.<explorer>` tooltips are now capable of displaying images from the `image` field (`http://url/to/image.png` or `file:///path/to/image.jpg`) of your `DataFrame`.
-   **Partial** towards audio data support
    -   `hover.core.explorer.<explorer>` tooltips are now capable of running audio playbacks from the `audio` field (`http://url/to/audio.mp3` or `file:///path/to/audio.wav`) of your `DataFrame`.

### Documentation

-   **Working** tutorials which use the awesome [`Juniper`](https://github.com/ines/juniper) to enable Binder-backed code blocks right in the documentation. Here are the materials coming up:
    -   tutorial 0: for the absolute beginner
    -   tutorial 1: vectorizer: what, why & how
    -   tutorial 2: 2d embedding: customization & resources
    -   tutorial 3: subsets: best practices with `raw` / `train` / `dev` / `test`
    -   [create an issue](https://github.com/phurwicz/hover/issues/new) to give us an idea :hugs:

### Backward Incompatibility

-   `hover.recipes.experimental.active_learning`: signature change of an argument: `vecnet_callback()` becomes `vecnet_callback(dataset, vectorizer)` for more flexibility

### Deprecation

-   `hover.core.explorer.BokehCorpusExplorer`: please use `hover.core.explorer.BokehCorpusFinder` instead.
    -   This is a naming change for consistency; the class functionality stays the same.
    -   The original class name will be removed in a future version.
-   `hover.core.neural.create_vector_net_from_module`: this makes more sense as a class method of `hover.core.neural.VectorNet`.
    -   The original function will be removed in a future version.
    -   The class method now takes either a loaded module or a string representing it. It (or the predecessor function) used to accept only the string form.

### Fixes

-   Cleaned up inconsistent logging format in the `hover.core` module.
    -   every class in `hover.core` now adopts `rich` and specifically `hover.core.Loggable` for logging.
        -   specified both foreground and background colors so that the text display clearly regardless of terminal settings.
