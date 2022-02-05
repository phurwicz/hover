> `Finder` is an `explorer` focused on **search**.
>
> :speedboat: It can help you select points using a **filter** based on search results.

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}
{!docs/snippets/markdown/component-tutorial.md!}

## **More Angles -> Better Results**

`Explorer`s other than `annotator` are specialized in finding additional insight to help us understand the data. Having them juxtaposed with `annotator`, we can label more accurately, more confidently, and even faster.

## **Preparation**

{!docs/snippets/markdown/dataset-prep.md!}

## **Filter Toggles**

When we use lasso or polygon select, we are describing a shape. Sometimes that shape is not accurate enough -- we need extra conditions to narrow down the data.

Just like `annotator`, `finder` has search widgets. But unlike `annotator`, `finder` has a **filter toggle** which can directly **intersect** *what we selected* with *what meets the search criteria*.

<pre data-executable>
{!docs/snippets/py/t5-0-finder-filter.txt!}
</pre><br>

Next to the search widgets is a checkbox. The filter will stay active as long as the checkbox is.

???+ info "How multiple checkboxes interact"
    Cool fact: the filter toggle works in conjunction with the **cumulative select toggle**.

    Actually, all of hover's toggles can be combined in an intuitive manner. In the later tutorials, we shall see multiple filters in action together.

## **Stronger Highlight for Search**

`finder` also colors data points based on search criteria, making them easier to find.

{!docs/snippets/markdown/jupyterlab-js-issue-lite.md!}

<pre data-executable>
{!docs/snippets/py/t5-1-finder-figure.txt!}
</pre><br>

{!docs/snippets/html/stylesheet.html!}
