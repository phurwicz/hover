> `hover` supports bulk-labeling audios through their URLs (which can be local).
>
> :bulb: Let's do a quickstart for audios and note what's different from texts.

{!docs/snippets/markdown/tutorial-required.md!}
{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}

## **Dataset for audios**

`hover` handles audios through their URL addresses. URLs are strings which can be easily stored, hashed, and looked up against. They are also convenient for rendering tooltips in the annotation interface.

Similarly to `SupervisableTextDataset`, we can build one for audios:

<pre data-executable>
{!docs/snippets/py/g1-0-dataset-audio.txt!}

{!docs/snippets/py/t0-0a-dataset-text-print.txt!}
</pre>

## **Vectorizer for audios**

We can follow a `URL -> content -> audio array -> vector` path.

<pre data-executable>
{!docs/snippets/py/g0-1-url-to-content.txt!}
</pre>

<pre data-executable>
{!docs/snippets/py/g1-1-url-to-audio.txt!}
</pre>

{!docs/snippets/markdown/wrappy-cache.md!}

<pre data-executable>
{!docs/snippets/py/g1-2-audio-vectorizer.txt!}
</pre>

## **Embedding and Plot**

This is exactly the same as in the quickstart, just switching to audio data:

<pre data-executable>
{!docs/snippets/py/t0-2-reduction.txt!}
</pre>

<pre data-executable>
{!docs/snippets/py/t0-3-simple-annotator.txt!}

{!docs/snippets/py/tz-bokeh-show-server.txt!}

{!docs/snippets/py/tz-bokeh-show-notebook.txt!}
</pre>

???+ note "What's special for audios?"
    **Tooltips**

    For text, the tooltip shows the original value.

    For audios, the tooltip embeds the audio based on URL.

    -   audios in the local file system shall be served through [`python -m http.server`](https://docs.python.org/3/library/http.server.html).
    -   they can then be accessed through `https://localhost:<port>/relative/path/to/file`.

    **Search**

    For text, the search widget is based on regular expressions.

    For audios, the search widget is based on vector cosine similarity.

    -   the `dataset` has remembered the `vectorizer` under the hood and passed it to the `annotator`.
    -   {== please [**let us know**](https://github.com/phurwicz/hover/issues/new) if you think there's a better way to search audios in this case. ==}
        -   dynamic time warping, due to its running time (> 10ms per pair for small 100x10 MFCC arrays), is too slow for search.
            -   we are experimenting with subsampled signals and pre-selected data points (by vector similarity, for example).


{!docs/snippets/html/stylesheet.html!}
