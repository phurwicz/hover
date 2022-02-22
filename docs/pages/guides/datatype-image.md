> `hover` supports bulk-labeling images through their URLs.
>
> Let's walk through an example showing how tooltips, display, and search work for images.
>
> **Make sure that you have seen the tutorial. This guide assumes that you know the basic usage of `SupervisableDataset`, vectorizers and recipes.**

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}

## **`SupervisableDataset` for Images**

As always, with a [`SupervisableDataset`](../../reference/core-dataset/#hover.core.dataset.SupervisableDataset)

<pre data-executable>
{!docs/snippets/py/g0-0-dataset-image.txt!}

{!docs/snippets/py/t0-0a-dataset-text-print.txt!}
</pre>

## **Vectorizer with URL Input**

A pre-trained embedding lets us group data points semantically.

In particular, let's define a `data -> embedding vector` function.

<pre data-executable>
{!docs/snippets/py/g0-1-url-to-content.txt!}

{!docs/snippets/py/g0-2-url-to-image.txt!}

{!docs/snippets/py/g0-3-image-vectorizer.txt!}
</pre>


{!docs/snippets/html/stylesheet.html!}
