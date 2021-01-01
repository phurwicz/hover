:hourglass: 0.4.0 - Upcoming
==================

### :tada: Features

-   [Added loading text](https://github.com/phurwicz/hover/commit/ee692e0dddfe186261e07c8008a6066b1a1fcd16) before each `hover.recipes.<recipe>` renders in BokehJS.
    -   displays an additional dot every a few (5) seconds.
    -   **potential security concern** displays traceback information if something breaks in the scope of the recipe.
    -   this can be useful, for example, when you are visiting a remote Bokeh server with no access to the logs.
-   **Partial** towards image data support
    -   `hover.core.explorer.<explorer>` [tooltips](https://github.com/phurwicz/hover/commit/41077fbe00258cc0cf07a1abdfbd0dcd324a3a66#diff-df680e588036004aa1b9a591492e1e52f0787ac5dedbd2a8f07064e714fb28a2R48) are now capable of displaying images from the `image` field (`http://url/to/image.png` or `file:///path/to/image.jpg`) of your `DataFrame`.
-   **Partial** towards audio data support
    -   `hover.core.explorer.<explorer>` [tooltips](https://github.com/phurwicz/hover/commit/41077fbe00258cc0cf07a1abdfbd0dcd324a3a66#diff-df680e588036004aa1b9a591492e1e52f0787ac5dedbd2a8f07064e714fb28a2R61) are now capable of running audio playbacks from the `audio` field (`http://url/to/audio.mp3` or `file:///path/to/audio.wav`) of your `DataFrame`.

### :book: Documentation

-   :hourglass: **Upcoming** tutorials which use the awesome [`Juniper`](https://github.com/ines/juniper) to enable Binder-backed code blocks right in the documentation. Here are the materials coming up:
    -   tutorial 0: for the absolute beginner
    -   tutorial 1: vectorizer: what, why & how
    -   tutorial 2: 2d embedding: customization & resources
    -   tutorial 3: subsets: best practices with `raw` / `train` / `dev` / `test`
    -   [create an issue](https://github.com/phurwicz/hover/issues/new) to give us an idea :hugs:

### :exclamation: Backward Incompatibility

-   [`hover.recipes.experimental.active_learning`](https://github.com/phurwicz/hover/blob/e5ef551445f99e7c2eae759066962d28fd48dbf1/hover/recipes/experimental.py#L45): signature change of an argument: [`vecnet_callback()` becomes `vecnet_callback(dataset, vectorizer)`](https://github.com/phurwicz/hover/commit/8391d76100870a13201c6a4be855fc178436b971#diff-b45bf51d118b093c078e2b2333eadd1c03d07a0801de85ecb40bc268b0a13288R76) for more flexibility

### :warning: Deprecation / Naming Changes

-   `hover.core.explorer.BokehCorpusExplorer`: please use `hover.core.explorer.BokehTextFinder` instead.
    -   This is a naming change for consistency; the class functionality stays the same.
    -   The original class name will be removed in a future version.
-   `hover.core.explorer.BokehCorpus<XXX>`: renamed to `hover.core.explorer.BokehText<XXX>` for consistency with `text/audio/image` features.
-   `hover.core.neural.create_vector_net_from_module`: this makes more sense as a [class method of `hover.core.neural.VectorNet`](https://github.com/phurwicz/hover/commit/8fcba7a46a69b3ae9c9dcadb4387cc9eaa711a09).
    -   The original function will be removed in a future version.
    -   The class method now takes either a loaded module or a string representing it. It (or the predecessor function) used to accept only the string form.

### :hammer: Fixes

-   Cleaned up inconsistent logging format in the `hover.core` module.
    -   every class in `hover.core` now adopts `rich` and specifically [`hover.core.Loggable`](https://github.com/phurwicz/hover/commits/main/hover/core/__init__.py) for logging.
        -   [specified both foreground and background colors](https://github.com/phurwicz/hover/commit/0477e6774176894e27b62e8c3c32c18352aad624#diff-802028d3406d84b35d490c3c0109000d83f883c18bea1d9719666fcd9a72f03a) so that the text display clearly regardless of terminal settings.
