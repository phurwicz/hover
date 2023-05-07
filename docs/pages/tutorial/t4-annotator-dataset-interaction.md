> `Annotator` is an `explorer` which provides a map of your data colored by labels.
>
> :speedboat: Let's walk through its components and how they interact with the `dataset`.
>
> -   {== You will find many of these components again in other `explorer`s. ==}

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}
{!docs/snippets/markdown/component-tutorial.md!}
{!docs/snippets/markdown/local-dependency.md!}
{!docs/snippets/markdown/local-dep-text.md!}
{!docs/snippets/markdown/local-dep-jupyter-bokeh.md!}

## **Preparation**

{!docs/snippets/markdown/dataset-prep.md!}

## **Scatter Plot: Semantically Similar Points are Close Together**

`hover` labels data points in bulk, which requires selecting groups of homogeneous data.

The core of the annotator is a scatter plot and labeling widgets:

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/tz-bokeh-notebook-common.txt!}

{!docs/snippets/py/tz-bokeh-notebook-remote.txt!}

{!docs/snippets/py/t4-0-annotator-basics.txt!}
</pre><br>

### **Select Points on the Plot**

On the right of the scatter plot, you can find tap, polygon, and lasso tools which can select data points.

### **View Tooltips with Mouse Hover**

Embeddings are helpful but rarely perfect. This is why we have tooltips that show the detail of each point on mouse hover, allowing us to inspect points, discover patterns, and come up with new labels on the fly.

### **Show & Hide Subsets**

Showing labeled subsets can tell you which parts of the data has been explored and which ones have not. With toggle buttons, you can turn on/off the display for any subset.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t4-1-annotator-subset-toggle.txt!}
</pre><br>

## **Make Consecutive Selections**

Ever selected multiple (non-adjacent) files in your file system using <kbd>Ctrl</kbd>/<kbd>Command</kbd>?

Similarly but more powerfully, you can make consecutive selections with a "keep selecting" option.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t4-2-annotator-selection-option.txt!}
</pre><br>

???+ info "Selection option values: what do they do?"
    Basic set operations on your old & new selection. [Quick intro here](https://www.geeksforgeeks.org/python-set-operations-union-intersection-difference-symmetric-difference/)

    -   `none`: the default, where a new selection `B` simply replaces the old one `A`.
    -   `union`: `A ∪ B`, the new selection gets unioned with the old one.
        -   this resembles the <kbd>Ctrl</kbd>/<kbd>Command</kbd> mentioned above.
    -   `intersection`: `A ∩ B`, the new selection gets intersected with the old one.
        -   this is particularly useful when going beyond simple 2D plots.
    -   `difference`: `A ∖ B`, the new selection gets subtracted from the old one.
        -   this is for de-selecting outliers.

## **Change Plot Axes**

`hover` supports dynamically choosing which embedding dimensions to use for your 2D plot. This becomes nontrivial, and sometimes very useful, when we have a 3D embedding (or higher):

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t0-2z-reduction-3d.txt!}

{!docs/snippets/py/t4-3-annotator-choose-axes.txt!}
</pre><br>

## **Text Search Widget: Include/Exclude**

Keywords or regular expressions can be great starting points for identifying a cluster of similar points based on domain expertise.

You may specify a *positive* regular expression to look for and/or a *negative* one to not look for.

The `annotator` will amplify the sizes of positive-match data points and shrink those of negative matches.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t4-4-annotator-search-box.txt!}
</pre><br>

### **Preview: Use Search for Selection in Finder**

In a particular kind of plot called `finder` (search it in the README!), the search widget can directly operate on your selection as a filter.

## **The Plot and The Dataset**

When we apply labels through the annotator plot, it's acutally the `dataset` behind the plot that gets immediately updated. The plot itself is not in direct sync with the dataset, which is a design choice for performance. Instead, we will use a trigger called `PUSH` for updating the data entries to the plot.

### **PUSH: Synchronize from Dataset to Plots**

Below is the full interface of the `dataset`, where you can find a green "Push" button:

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t4-5-dataset-view.txt!}
</pre>

In a built-in `recipe`, the "Push" button will update the latest data to every `explorer` linked to the `dataset`.

{!docs/snippets/html/stylesheet.html!}
