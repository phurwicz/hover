# CHANGELOG

## 0.6.0 - In Progress

### :tada: Features Added

-   all `BokehBaseExplorer` subclasses
    -   Selections can now be made cumulatively. Tap on multiple points to view or label at once, without the overhead of re-plotting in between.
        -   this option is invoked through a checkbox toggle.
        -   By default, built-in recipes link the toggle between all explorers in the recipe.

-   `VectorNet`
    -   now has widgets for configuring training hyperparameters.
        -   currently only supports changing epochs.
        -   will support changing learning rate and momentum.
    -   added a method prepare_loader() that takes `SupervisableDataset` and returns a torch `DataLoader`.

-   `MultiVectorNet` **new class**
    -   makes use of multiple VectorNets trained simutaneously, inspired by the [Coteaching research](https://arxiv.org/abs/1804.06872).

### :exclamation: Backward Incompatibility

-   `active_learning` **signature change**
    -   no longer takes `vectorizer` as an input. Instead, `VectorNet`/`MultiVectorNet` produced by `vecnet_callback` will handle the vectorization of raw input data.

## 0.5.0 - Apr 30, 2021

### :tada: Features Added

-   `SupervisableDataset`
    -   Added [import/export from/to pandas dataframes](https://github.com/phurwicz/hover/commit/e4f1a27f66a79c031f0163f14896bc30bbaff567);
    -   Added [a table for viewing selected data](https://github.com/phurwicz/hover/commit/2f442a51f8e8c04695ad6e185a90eea157f08689) to the visual interface.
    -   Added an "Export" button -- this is taken from `BokehDataAnnotator`.

-   `BokehDataFinder`
    -   Search criteria can now be used to [filter data point selections through a checkbox toggle](https://github.com/phurwicz/hover/commit/5b8fb17f50a3c36726d4665d1cf1253ba4d5f2f9).

-   `BokehSoftLabelExplorer`
    -   Soft scores can now be used to [filter data point selections through a checkbox toggle](https://github.com/phurwicz/hover/commit/5b8fb17f50a3c36726d4665d1cf1253ba4d5f2f9).

-   Intersecting Filters
    -   The two filters mentioned above can be in effect simultaneously, taking a set intersection.

-   Recipe: `active_learning`
    -   Added an [interpolation of decision boundaries]((https://github.com/phurwicz/hover/commit/dc5851514799825a66911a266a9abb2066e92078)) between 2D embeddings of the input and output.

-   `VectorNet`
    -   It is now possible to [configure whether or not to backup the model state dict](https://github.com/phurwicz/hover/commit/fe150b34a89993e8d7df4c6c76a2b729ff411268).

### :exclamation: Feature Removed

-   A few JavaScript callbacks are converted to Python and will no longer work in static HTML:
    -   search widget responses (i.e. glyph color/size changes) in all `explorer`s;
    -   synchronization of data point selections between `explorer`s, e.g. in the `linked_annotator` recipe.

-   `BokehDataAnnotator`
    -   Removed the "Export" button -- it is now with `SupervisableDataset` instead.

### :hammer: Fixes

-   Keyword arguments to recipes (`simple_annotator`, for example) are now [correctly forwarded to Bokeh figures](https://github.com/phurwicz/hover/commit/5c4e6b46140fcbec974b0ee88a2dc2175f2f3c50).
    -   Note that figure tooltips (the info box that show up upon mouse hover over data points) will *append to*, instead of replace, the built-in tooltips.

-   `active_learning`'s dev set during model re-train will now [fall back to using the train set](https://github.com/phurwicz/hover/commit/fdc1217f3b7c3428e7fa831b1d003df368ab568f) if the dev set is empty.

## 0.4.1 - Jan 31, 2021

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
