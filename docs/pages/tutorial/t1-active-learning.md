> The most common usage of `hover` is through built-in `recipe`s like in the quickstart.
>
> :ferris_wheel: Let's explore another `recipe` -- an active learning example.

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}

## **Fundamentals**

Hover `recipe`s are functions that take a `SupervisableDataset` and return an annotation interface.

The `SupervisableDataset` is assumed to have some data and embeddings.

## **Recap: Data & Embeddings**

Let's preprare a dataset with embeddings. This is exactly the same as in the [quickstart](../t0-quickstart/):

<pre data-executable>
{!docs/snippets/py/t0-0-dataset-text.txt!}
</pre><br>

<pre data-executable>
{!docs/snippets/py/t0-1-vectorizer.txt!}
</pre><br>

<pre data-executable>
{!docs/snippets/py/t0-2-reduction.txt!}
</pre>


## **Recipe-Specific Ingredient**

Each recipe has different functionalities and potentially different signature.

To utilize active learning, we need to specify how to get a model in the loop.

`hover` considers the `vectorizer` as a "frozen" embedding and follows up with a neural network, which infers its own dimensionality from the vectorizer and the output classes.

-   This architecture named [`VectorNet`](../../reference/core-neural/#hover.core.neural.VectorNet) is the (default) basis of active learning in `hover`.

??? info "Custom models"
    It is possible to use a model other than `VectorNet` or its subclass.

    You will need to implement the following methods with the same signatures as `VectorNet`:

    -   [`train`](../../reference/core-neural/#hover.core.neural.VectorNet.train)
    -   [`save`](../../reference/core-neural/#hover.core.neural.VectorNet.save)
    -   [`predict_proba`](../../reference/core-neural/#hover.core.neural.VectorNet.predict_proba)
    -   [`prepare_loader`](../../reference/core-neural/#hover.core.neural.VectorNet.prepare_loader)
    -   [`manifold_trajectory`](../../reference/core-neural/#hover.core.neural.VectorNet.manifold_trajectory)

<pre data-executable>
{!docs/snippets/py/t1-0-vecnet-callback.txt!}

{!docs/snippets/py/t1-0a-vecnet-callback-print.txt!}
</pre>

Note how the callback dynamically takes `dataset.classes`, which means the model architecture will adapt when we add classes during annotation.


## :sparkles: **Apply Labels**

Now we invoke the `active_learning` recipe.

??? tip "Tips: how recipes work programmatically"
    In general, a `recipe` is a function taking a `SupervisableDataset` and other arguments based on its functionality.

    Here are a few common recipes:

    === "active_learning"

        ::: hover.recipes.experimental.active_learning
            rendering:
              show_root_heading: false
              show_root_toc_entry: false

    === "simple_annotator"

        ::: hover.recipes.stable.simple_annotator
            rendering:
              show_root_heading: false
              show_root_toc_entry: false

    === "linked_annotator"

        ::: hover.recipes.stable.linked_annotator
            rendering:
              show_root_heading: false
              show_root_toc_entry: false

    The recipe returns a `handle` function which `bokeh` can use to visualize an annotation interface in multiple settings.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t1-1-active-learning.txt!}
</pre>

???+ tip "Tips: annotation interface with multiple plots"
    ??? example "Video guide: leveraging linked selection"
        <iframe width="560" height="315" src="https://www.youtube.com/embed/TIwBlCH9YHw" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

    ???+ example "Video guide: active learning"
        <iframe width="560" height="315" src="https://www.youtube.com/embed/hRIn3r7ovQ8" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

    ??? info "Text guide: active learning"
        Inspecting model predictions allows us to

        -   get an idea of how the current set of annotations will likely teach the model.
        -   locate the most valuable samples for further annotation.

{!docs/snippets/html/stylesheet.html!}
