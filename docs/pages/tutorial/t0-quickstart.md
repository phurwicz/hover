> Welcome to the minimal guide of `hover`!
>
> :sunglasses: Let's label some data and call it a day.

{!docs/snippets/html/stylesheet.html!}

## **Ingredient 1 / 3: Some Data**

Suppose that we have a list of data entries, each in the form of a dictionary.

We can first create a [`SupervisableDataset`](../../reference/core-dataset/#hover.core.dataset.SupervisableDataset) based on those entries:

<pre data-executable>
{!docs/snippets/py/t0-0-dataset-text.txt!}
</pre><br>


## **Ingredient 2 / 3: Vectorizer**

To put our dataset sensibly on a 2-D "map", we will use a vectorizer for feature extraction, and then perform dimensionality reduction.<br>

Here's one way to define a vectorizer:

<pre data-executable>
{!docs/snippets/py/t0-1-vectorizer.txt!}
</pre><br>


## **Ingredient 3 / 3: Reduction**

The dataset has built-in high-level support for dimensionality reduction. <br>
Currently we can use [umap](https://umap-learn.readthedocs.io/en/latest/) or [ivis](https://bering-ivis.readthedocs.io/en/latest/).

??? info "Optional dependencies"
    The corresponding libraries do not ship with hover by default, and may need to be installed:

    -   for umap: `pip install umap-learn`
    -   for ivis: `pip install ivis[cpu]` or `pip install ivis[gpu]`

    `umap-learn` is installed in this demo environment.

<pre data-executable>
{!docs/snippets/py/t0-2-reduction.txt!}
</pre><br>


## :sparkles: **Apply Labels**

Now we are ready to visualize and annotate!

???+ tip "Basic tips"
    There should be a `SupervisableDataset` board on the left and an `BokehDataAnnotator` on the right.

    The `SupervisableDataset` comes with a few buttons:

    -   `push`: push `Dataset` updates to the bokeh plots.
    -   `commit`: add data entries selected in the `Annotator` to a specified subset.
    -   `dedup`: deduplicate across subsets (keep the last entry).

    The `BokehDataAnnotator` comes with a few buttons:

    -   `raw`/`train`/`dev`/`test`: choose which subsets to display.
    -   `apply`: apply the `label` input to the selected points in the `raw` subset only.
    -   `export`: save your data (all subsets) in a specified format.

??? info "Best practices"
    We've essentially put the data into neighborboods based on the vectorizer, but the quality, or the homogeneity of labels, of such neighborhoods can vary.

    -   hover over any data point to see its tooltip.
    -   take advantage of different selection tools to apply labels at appropriate scales.
    -   the search widget might turn out useful.
        -    note that it does not select points but highlights them.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t0-3-simple-annotator.txt!}
</pre><br>

{!docs/snippets/html/juniper.html!}
