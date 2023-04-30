> `hover` supports bulk-labeling images through their URLs (which can be local).
>
> :bulb: Let's do a quickstart for images and note what's different from texts.

{!docs/snippets/markdown/tutorial-required.md!}
{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}

## **Dataset for Images**

`hover` handles images through their URL addresses. URLs are strings which can be easily stored, hashed, and looked up against. They are also convenient for rendering tooltips in the annotation interface.

Similarly to `SupervisableTextDataset`, we can build one for images:

<pre data-executable>
{!docs/snippets/py/g0-0-dataset-image.txt!}

{!docs/snippets/py/t0-0a-dataset-text-print.txt!}
</pre>

## **Vectorizer for Images**

We can follow a `URL -> content -> image object -> vector` path.

<pre data-executable>
{!docs/snippets/py/g0-1-url-to-content.txt!}
</pre>

<pre data-executable>
{!docs/snippets/py/g0-2-url-to-image.txt!}
</pre>

{!docs/snippets/markdown/wrappy-cache.md!}

<pre data-executable>
{!docs/snippets/py/g0-3-image-vectorizer.txt!}
</pre>

## **Embedding and Plot**

This is exactly the same as in the quickstart, just switching to image data:

<pre data-executable>
{!docs/snippets/py/t0-2-reduction.txt!}
</pre>

<pre data-executable>
{!docs/snippets/py/t0-3-simple-annotator.txt!}

{!docs/snippets/py/tz-bokeh-show-server.txt!}

{!docs/snippets/py/tz-bokeh-show-notebook.txt!}
</pre>

???+ note "What's special for images?"
    **Tooltips**

    For text, the tooltip shows the original value.

    For images, the tooltip embeds the image based on URL.

    -   images in the local file system shall be served through [`python -m http.server`](https://docs.python.org/3/library/http.server.html).
    -   they can then be accessed through `https://localhost:<port>/relative/path/to/file`.

    **Search**

    For text, the search widget is based on regular expressions.

    For images, the search widget is based on vector cosine similarity.

    -   the `dataset` has remembered the `vectorizer` under the hood and passed it to the `annotator`.
    -   {== please [**let us know**](https://github.com/phurwicz/hover/issues/new) if you think there's a better way to search images in this case. ==}


{!docs/snippets/html/stylesheet.html!}
