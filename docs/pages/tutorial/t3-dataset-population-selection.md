> `SupervisableDataset` holds your data throughout the labeling process.
>
> :speedboat: Let's take a look at its core mechanisms.

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}
{!docs/snippets/markdown/component-tutorial.md!}

## **Data Subsets**

We place unlabeled data and labeled data in different subsets: "raw", "train", "dev", and "test". Unlabeled data start from the "raw" subset, and can be transferred to other subsets after it gets labeled.

`SupervisableDataset` uses a "population table", `dataset.pop_table`, to show the size of each subset:

<pre data-executable>
{!docs/snippets/py/tz-dataset-text-full.txt!}
</pre><br>

<pre data-executable>
{!docs/snippets/py/tz-bokeh-notebook.txt!}

{!docs/snippets/py/t3-0-dataset-population-table.txt!}
</pre><br>

### **Transfer Data Between Subsets**

`COMMIT` and `DEDUP` are the mechanisms that `hover` uses to transfer data between subsets.

-   `COMMIT` copies selected points (to be discussed later) to a destination subset
    -   labeled-raw-only: `COMMIT` automatically detects which points are in the raw set with a valid label. Other points will not get copied.
    -   keep-last: you can commit the same point to the same subset multiple times and the last copy will be kept. This can be useful for revising labels before `DEDUP`.
-   `DEDUP` removes duplicates (identified by feature value) across subsets
    -   priority rule: test > dev > train > raw, i.e. test set data always gets kept during deduplication

???+ info "FAQ"
    ??? help "Why does COMMIT only work on the raw subset?"
        Most selections will happen through plots, where different subsets are on top of each other. This means selections can contain both unlabeled and labeled points.

        Way too often we find ourselves trying to view both the labeled and the unlabeled, but only moving the unlabeled "raw" points. So it's handy that COMMIT picks those points only.

These mechanisms correspond to buttons in `hover`'s annotation interface, which you have encountered in the quickstart:

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t3-1-dataset-commit-dedup.txt!}
</pre><br>

Of course, so far we have nothing to move, because there's no data selected. We shall now discuss selections.

## **Selection**

`hover` labels data points in bulk, which requires selecting groups of homogeneous data, i.e. semantically similar or going to have the same label. Being able to skim through what you selected gives you confidence about homogeneity.

Normally, selection happens through a plot (`explorer`), as we have seen in the quickstart. For the purpose here, we will "cheat" and assign the selection programmatically:

<pre data-executable>
{!docs/snippets/py/t3-2-dataset-selection-table.txt!}
</pre><br>

### **Edit Data Within a Selection**

Often the points selected are not perfectly homogeneous, i.e. some outliers belong to a different label from the selected group overall. It would be helpful to `EVICT` them, and `SupervisableDataset` has a button for it.

Sometimes you may also wish to edit data values on the fly.  In hover this is called `PATCH`, and there also is a button for it.

-   by default, labels can be edited but feature values cannot.

Let's plot the forementioned buttons along with the selection table. Toggle any number of rows in the table, then click the button to `EVICT` or `PATCH` those rows:

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t3-3-dataset-evict-patch.txt!}
</pre><br>


{!docs/snippets/html/stylesheet.html!}
