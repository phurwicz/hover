0.4.0 - Upcoming
==================

### Features
- Added loading text before each `hover.recipes.<recipe>` renders in BokehJS.
-   **Partial** image data support
    -   `hover.core.explorer.<explorer>` tooltips are now capable of displaying images from the `image` field (URL) of your `DataFrame`.
-   **Partial** audio data support
    -   `hover.core.explorer.<explorer>` tooltips are now capable of running audio playbacks from the `audio` field (URL) of your `DataFrame`.

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
