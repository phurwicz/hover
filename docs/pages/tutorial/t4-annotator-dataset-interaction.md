> `Annotator` is an `explorer` which provides a map of your data colored by labels.
>
> :speedboat: Let's walk through its visual components and how they interact with the `SupervisableDataset`.

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}
{!docs/snippets/markdown/component-tutorial.md!}

## **Scatter Plot: Semantically Similar Points are Close Together**

`hover` labels data points in bulk, which requires selecting groups of homogeneous data.

Embeddings are often helpful but not perfect. This is why we have tooltips that show the detail of each point on mouse hover:

<pre data-executable>
{!docs/snippets/py/t0-0-dataset-text.txt!}

{!docs/snippets/py/t0-1-vectorizer.txt!}

{!docs/snippets/py/t0-2-reduction.txt!}

{!docs/snippets/py/t4-0-annotator-figure.txt!}
</pre><br>

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

## **PUSH: synchronize from dataset to plots**

The plotted data is not in direct sync with the underlying dataset, which is a design choice for performance. Instead we will use a trigger called `PUSH` for this synchronization.

Below is the interface from the quickstart, where you can find a green "Push" button:

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t0-3-simple-annotator.txt!}
</pre>

{!docs/snippets/html/stylesheet.html!}
