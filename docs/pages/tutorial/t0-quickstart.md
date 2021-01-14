> Welcome to the minimal guide of `hover`!
>
> :sunglasses: Let's label some data and call it a day.

{!docs/snippets/html/stylesheet.html!}

## **Ingredient 1 / 3: Some Data**

Suppose that we have a list of data entries, each in the form of a dictionary:

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

??? warning "Known issue"
    {== If you are running this code block on this documentation page: ==}

    -   JavaScript output (which contains the visualization) will fail to render due to JupyterLab's security restrictions.
    -   please run this tutorial locally to view the output.

    ??? help "Advanced: help wanted"
        Some context:

        -   the code blocks here are embedded using [Juniper](https://github.com/ines/juniper).
        -   the environment is configured in the [Binder repo](https://github.com/phurwicz/hover-binder).

        What we've tried:

        -   1 [Bokeh's extension with JupyterLab](https://github.com/bokeh/jupyter_bokeh)
            -   1.1 cannot render the Bokeh plots remotely with `show(handle)`, with or without the extension
                -   1.1.1 JavaScript console suggests that `bokeh.main.js` would fail to load.
        -   2 [JavaScript magic cell](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-javascript)
            -   2.1 such magic is functional in a custom notebook on the Jupyter server.
            -   2.2 such magic is blocked by JupyterLab if ran on the documentation page.

        Tentative clues:

        -   2.1 & 2.2 suggests that somehow JupyterLab behaves differently between Binder itself and Juniper.
        -   Juniper by default [trusts the cells](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-javascript).
        -   making Javascript magic work on this documentation page would be a great step.

<pre data-executable>
{!docs/snippets/py/t0-3-simple-annotator.txt!}
</pre><br>

{!docs/snippets/html/juniper.html!}
