> Welcome to the simplest use case of `hover`!
>
> :sunglasses: Let's say we want to label some data and call it a day.

{!docs/snippets/html/stylesheet.html!}

## **Ingredient 1 / 3: Some Data**

Suppose that we have a list of data entries, each in the form of a dictionary.

We can first create a [`SupervisableDataset`](../../reference/core-dataset/#hover.core.dataset.SupervisableDataset) based on those entries:

<pre data-executable>
{!docs/snippets/py/t0-0-dataset-text.txt!}
</pre>

???+ info "FAQ"
    ??? help "What if I have multiple features?"
        `feature_key` refers to the field that will be vectorized later on, which can be a JSON that encloses multiple features.

        For example, suppose our data entries look like this:
        ```python
        {"f1": "foo", "f2": "bar", "non_feature": "abc"}
        ```

        We can put `f1` and `f2` in a JSON and convert the entries like this:
        ```python
        # could also keep f1 and f2 around
        {'feature': '{"f1": "foo", "f2": "bar"}', 'non_feature': 'abc'}
        ```

    ??? help "Can I use audio or image data?"
        In the not-too-far future, yes!

        Some mechanisms can get tricky with audios/images, but we are working on it:

        -   display tooltips: as of 0.4.0, [tooltips are supported in the low-level APIs](https://github.com/phurwicz/hover/blob/main/hover/core/explorer/local_config.py).
        -   [search and highlight](../../reference/core-explorer/#hover.core.explorer.feature.BokehForAudio.activate_search): pending (and open to contributions!)
        -   high-level API like `SupervisableImageDataset`: pending



## **Ingredient 2 / 3: Vectorizer**

To put our dataset sensibly on a 2-D "map", we will use a vectorizer for feature extraction, and then perform dimensionality reduction.

Here's one of many ways to define a vectorizer:

<pre data-executable>
{!docs/snippets/py/t0-1-vectorizer.txt!}
</pre>

???+ tip "Tips"
    ??? example "Caching"
        `dataset` by itself stores the original features but not the corresponding vectors.

        To avoid vectorizing the same feature again and again, we could simply do:
        ```python
        from functools import cache

        @cache
        def vectorizer(feature):
            # put code here
        ```

        If you'd like to limit the size of the cache, something like `@lru_cache(maxsize=10000)` could help.

        Check out [functools](https://docs.python.org/3/library/functools.html) for more options.

    ??? example "Vectorizing multiple features"
        Suppose we have multiple features enclosed in a JSON:

        ```python
        # could also keep f1 and f2 around
        {'feature': '{"f1": "foo", "f2": "bar"}', 'non_feature': 'abc'}
        ```

        Also, suppose we have individual vectorizers likes this:
        ```python
        def vectorizer_1(feature_1):
            # put code here

        def vectorizer_2(feature_2):
            # put code here
        ```

        Then we can define a composite vectorizer:
        ```python
        import json
        import numpy as np

        def vectorizer(feature_json):
            data_dict = json.loads(feature_json)
            vectors = []
            for field, func in [
                ("f1", vectorizer_1),
                ("f2", vectorizer_2),
            ]:
                vectors.append(func(data_dict[field]))

            return np.concatenate(vectors)
        ```


## **Ingredient 3 / 3: Reduction**

The dataset has built-in high-level support for dimensionality reduction.

Currently we can use [umap](https://umap-learn.readthedocs.io/en/latest/) or [ivis](https://bering-ivis.readthedocs.io/en/latest/).

??? info "Optional dependencies"
    The corresponding libraries do not ship with hover by default, and may need to be installed:

    -   for umap: `pip install umap-learn`
    -   for ivis: `pip install ivis[cpu]` or `pip install ivis[gpu]`

    `umap-learn` is installed in this demo environment.

<pre data-executable>
{!docs/snippets/py/t0-2-reduction.txt!}
</pre>


## :sparkles: **Apply Labels**

Now we are ready to visualize and annotate!

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t0-3-simple-annotator.txt!}
</pre>

???+ tip "Tips: annotation interface basics"
    ???+ example "Video guide"
        <iframe width="560" height="315" src="https://www.youtube.com/embed/WYN2WduzJWg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

    ??? info "Text guide"
        There should be a `SupervisableDataset` board on the left and an `BokehDataAnnotator` on the right, each with a few buttons.

        === "SupervisableDataset"
            -   `push`: push `Dataset` updates to the bokeh plots.
            -   `commit`: add data entries selected in the `Annotator` to a specified subset.
            -   `dedup`: deduplicate across subsets by `feature` (last in gets kept).

        === "BokehDataAnnotator"
            -   `raw`/`train`/`dev`/`test`: choose which subsets to display or hide.
            -   `apply`: apply the `label` input to the selected points in the `raw` subset only.
            -   `export`: save your data (all subsets) in a specified format.

        We've essentially put the data into neighborboods based on the vectorizer, but the quality (homogeneity of labels) of such neighborhoods can vary.

        -   hover over any data point to see its tooltip.
        -   take advantage of different selection tools to apply labels at appropriate scales.
        -   the search widget might turn out useful.
            -    note that it does not select points but highlights them.

{!docs/snippets/html/juniper.html!}
