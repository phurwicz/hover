> `Annotator` is an `explorer` which provides a map of your data colored by labels.
>
> :speedboat: Let's walk through its components and how they interact with the `dataset`.
>
> -   {== You will find many of these components again in other `explorer`s. ==}

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}
{!docs/snippets/markdown/component-tutorial.md!}

## **Preparation**

{!docs/snippets/markdown/dataset-prep.md!}

## **Scatter Plot: Semantically Similar Points are Close Together**

`hover` labels data points in bulk, which requires selecting groups of homogeneous data.

The core of the annotator is a scatter plot and labeling widgets:

<pre data-executable>
{!docs/snippets/py/t4-0-annotator-basics.txt!}
</pre><br>

Embeddings are helpful but rarely perfect. This is why we have tooltips that show the detail of each point on mouse hover, allowing us to inspect points, discover patterns, and come up with new labels on the fly.

### **Show & Hide Subsets**

Showing labeled subsets can tell you which parts of the data has been explored and which ones have not. With toggle buttons, you can turn on/off the display for any subset.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t4-1-annotator-subset-toggle.txt!}
</pre><br>

## **Selection Tools: Tap, Polygon, Lasso**

On the right of the scatter plot, you can find selection tools. Feel free to play with them for a bit.

### **Making Cumulative Selections**

Ever selected multiple files in your file system using <kbd>Ctrl</kbd>/<kbd>Command</kbd>?

Similarly in `hover`, you can make cumulative selections by toggleing a checkbox.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t4-2-annotator-selection-option.txt!}
</pre><br>

## **Text Search Widget: Include/Exclude**

Keywords or regular expressions can be great starting points for identifying a cluster of similar points based on domain expertise.

You may specify a *positive* regular expression to look for and/or a *negative* one to not look for.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t4-3-annotator-search-box.txt!}
</pre><br>

### **Preview: Use Search for Selection in Finder**

In a particular kind of plots called `Finder` (see later in the tutorials), the search widget can directly operate on your selection as a filter.

## **The Plot and The Dataset**

When we apply labels through the annotator plot, it's acutally the `dataset` behind the plot that gets immediately updated. The plot itself is not in direct sync with the dataset, which is a design choice for performance. Instead, we will use a trigger called `PUSH` for updating the data entries to the plot.

### **PUSH: synchronize from dataset to plots**

Below is the full interface of the `dataset`, where you can find a green "Push" button:

<pre data-executable>
{!docs/snippets/py/t4-4-dataset-view.txt!}
</pre>

In a built-in `recipe`, the "Push" button will update the latest data to every `explorer` linked to the `dataset`.

{!docs/snippets/html/stylesheet.html!}
