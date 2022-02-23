> `hover` filters can stack together.
>
> :speedboat: This makes selections incredibly powerful.

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}
{!docs/snippets/markdown/component-tutorial.md!}

## **Preparation**

{!docs/snippets/markdown/dataset-prep.md!}

## **Soft-Label Explorer**

Active learning works by predicting labels and scores (i.e. soft labels) and utilizing that prediction. An intuitive way to plot soft labels is to color-code labels and use opacity ("alpha" by `bokeh` terminology) to represent scores.

`SoftLabelExplorer` delivers this functionality:

<pre data-executable>
{!docs/snippets/py/tz-bokeh-notebook.txt!}

{!docs/snippets/py/t6-0-softlabel-figure.txt!}
</pre><br>

## **Filter Selection by Score Range**

Similarly to `finder`, a `softlabel` plot has its own selection filter. The difference lies in the filter condition:

<pre data-executable>
{!docs/snippets/py/t6-1-softlabel-filter.txt!}
</pre><br>

## **Linked Selections & Joint Filters**

When we plot multiple `explorer`s for the same `dataset`, it makes sense to synchronize selections between those plots. `hover` recipes take care of this synchronization.

-   :tada: This also works with cumulative selections. Consequently, the cumulative toggle is synchronized too.

Since each filter is narrowing down the selections we make, joint filters is just set intersection, extended

-   from two sets (original selection + filter)
-   to N sets (original selection + filter A + filter B + ...)

The [`active_learning` recipe]((../t1-active-learning/)) is built of `softlabel + annotator + finder`, plus a few widgets for iterating the model-in-loop.

In the next tutorial(s), we will see more recipes taking advantage of linked selections and joint filters. Powerful indeed!

{!docs/snippets/html/stylesheet.html!}
