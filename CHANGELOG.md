# CHANGELOG

## 0.4.1 - Projected Jan 31, 2021

### :hammer: Fixes

-   Label -> Glyph legends are removed from `BokehDataAnnotator` and `BokehSoftLabelExplorer`.
    -   instead, `SupervisableDataset` now keeps track of labels and their corresponding colors consistently with explorers which use colors based on labels.
    -   context for those who might be interested: [the tie between legends and renderers](https://docs.bokeh.org/en/latest/docs/user_guide/annotations.html#legends) makes legends hard to read when its renderer's glyphs vary a lot. `BokehDataAnnotator` and `BokehSoftLabelExplorer` dynamically update glyphs and fall into this scenario.

-   `BokehSoftLabelExplorer` now "smartly" calculates `fill_alpha`.
    -   it does so through the mean value and standard deviation of soft scores.

-   Clicking the `Commit` button on the interface of `SupervisableDataset` used to fail to reflect changes in the population table when a new class shows up.
    -   this is now resolved. Clicking either `Commit` or `Dedup` will reflect any population changes.

## 0.4.0 - Jan 3, 2021

### :tada: Features

-   [Added loading text](https://github.com/phurwicz/hover/commit/ee692e0dddfe186261e07c8008a6066b1a1fcd16) before each `hover.recipes.<recipe>` renders in BokehJS.
    -   displays an additional dot every a few (5) seconds.
    -   **potential security concern** displays traceback information if something breaks in the scope of the recipe.
    -   this can be useful, for example, when you are visiting a remote Bokeh server with no access to the logs.

-   **Partial** towards image data support
    -   `hover.core.explorer.<explorer>` [tooltips](https://github.com/phurwicz/hover/commit/41077fbe00258cc0cf07a1abdfbd0dcd324a3a66#diff-df680e588036004aa1b9a591492e1e52f0787ac5dedbd2a8f07064e714fb28a2R48) are now capable of displaying images from the `image` field (`http://url/to/image.png` or `file:///path/to/image.jpg`) of your `DataFrame`.

-   **Partial** towards audio data support
    -   `hover.core.explorer.<explorer>` [tooltips](https://github.com/phurwicz/hover/commit/41077fbe00258cc0cf07a1abdfbd0dcd324a3a66#diff-df680e588036004aa1b9a591492e1e52f0787ac5dedbd2a8f07064e714fb28a2R61) are now capable of running audio playbacks from the `audio` field (`http://url/to/audio.mp3` or `file:///path/to/audio.wav`) of your `DataFrame`.

### :exclamation: Backward Incompatibility

-   [`hover.recipes.experimental.active_learning`](https://github.com/phurwicz/hover/blob/e5ef551445f99e7c2eae759066962d28fd48dbf1/hover/recipes/experimental.py#L45): signature change of an argument: [`vecnet_callback()` becomes `vecnet_callback(dataset, vectorizer)`](https://github.com/phurwicz/hover/commit/8391d76100870a13201c6a4be855fc178436b971#diff-b45bf51d118b093c078e2b2333eadd1c03d07a0801de85ecb40bc268b0a13288R76) for more flexibility
-   [`hover.core.explorer.reset_figure()`] has been removed after a period of deprecation.

### :warning: Deprecation / Naming Changes

-   `hover.core.explorer.BokehCorpusExplorer`: please use `hover.core.explorer.BokehTextFinder` instead.
    -   This is a naming change for consistency; the class functionality stays the same.
    -   The original class name will be removed in a future version.

-   `hover.core.explorer.BokehCorpus<XXX>`: renamed to `hover.core.explorer.BokehText<XXX>` for consistency with `text/audio/image` features.

-   `hover.core.neural.create_vector_net_from_module`: this makes more sense as a [class method of `hover.core.neural.VectorNet`](https://github.com/phurwicz/hover/commit/8fcba7a46a69b3ae9c9dcadb4387cc9eaa711a09).
    -   The original function will be removed in a future version.
    -   The class method now takes either a loaded module or a string representing it. It (or the predecessor function) used to accept only the string form.

### :hammer: Fixes

-   [Legend](https://docs.bokeh.org/en/latest/docs/user_guide/annotations.html#legends) icons used to be broken when a legend item corresponded to multiple glyphs. This is now [resolved](https://github.com/phurwicz/hover/commit/e8ad3ff896585fc838bfd0da9cd7bd9ec1b6a17d) at the cost of dropping renderer references from the legend item.
    -   the context is that a `Bokeh` legend item shows its own icon by combining the glyphs of its renderers. The combined icon can be confusing when the glyphs' filling colors add up this way, even if they are the same color.

-   [Active learning](https://github.com/phurwicz/hover/commits/main/hover/recipes/experimental.py): [added a `model.save()` step](https://github.com/phurwicz/hover/commit/f926ac0f0cdfbbc554047a769768fb0673ec28ed) in the training callback so that the model resumes its progress on each callback, rather than starting over

-   Cleaned up inconsistent logging format in the `hover.core` module.
    -   every class in `hover.core` now adopts `rich` and specifically [`hover.core.Loggable`](https://github.com/phurwicz/hover/commits/main/hover/core/__init__.py) for logging.
        -   [specified both foreground and background colors](https://github.com/phurwicz/hover/commit/0477e6774176894e27b62e8c3c32c18352aad624#diff-802028d3406d84b35d490c3c0109000d83f883c18bea1d9719666fcd9a72f03a) so that the text display clearly regardless of terminal settings.
