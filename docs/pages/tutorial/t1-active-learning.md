> The most common usage of `hover` is through built-in `recipe`s like in the quickstart.
>
> :ferris_wheel: Let's explore another `recipe` -- an active learning example.

{!docs/snippets/html/stylesheet.html!}

## **Ingredient 1 ~ 3 / 4: Data, Vectorizer, Reduction**

This is exactly the same as in the [quickstart](../t0-quickstart/):

<pre data-executable>
{!docs/snippets/py/t0-0-dataset-text.txt!}
</pre><br>

<pre data-executable>
{!docs/snippets/py/t0-1-vectorizer.txt!}
</pre><br>

<pre data-executable>
{!docs/snippets/py/t0-2-reduction.txt!}
</pre>


## **Ingredient 4 / 4: Model Callback**

To utilize active learning, we need to specify how to get a model in the loop.

`hover` considers the `vectorizer` as a "frozen" embedding and follows up with a neural network, which infers its own dimensionality from the vectorizer and the output classes.

-   This architecture named [`VectorNet`](../../reference/core-neural/#hover.core.neural.VectorNet) is the (default) basis of active learning in `hover`.

??? info "Custom models"
    It is possible to use a model other than `VectorNet` or its subclass.

    Simply implement the following methods with the same signatures as `VectorNet`:

    -   [`train`](../../reference/core-neural/#hover.core.neural.VectorNet.train)
    -   [`save`](../../reference/core-neural/#hover.core.neural.VectorNet.save)
    -   [`predict_proba`](../../reference/core-neural/#hover.core.neural.VectorNet.predict_proba)

<pre data-executable>
{!docs/snippets/py/t1-0-vecnet-callback.txt!}
</pre>

Note how the callback dynamically takes `dataset.classes`, which means the model architecture will adapt when we add classes during annotation.


## :sparkles: **Recipe Time**

Now we invoke the `active_learning` recipe.

Inspecting model predictions allows us to

-   get an idea of how the current set of annotations will likely teach the model.
-   locate the most valuable samples for further annotation.

???+ tip "Basic tips"
    In general, a `recipe` is a function taking a `SupervisableDataset` and other arguments based on its functionality.

    Here are the most recommended recipes::

    === "active_learning"
        ::: hover.recipes.experimental.active_learning
            rendering:
              show_root_heading: true
              show_root_toc_entry: false
              heading_level: 4

    === "simple_annotator"
        ::: hover.recipes.stable.simple_annotator
            rendering:
              show_root_heading: true
              show_root_toc_entry: false
              heading_level: 4

    === "snorkel_crosscheck"
        ::: hover.recipes.experimental.snorkel_crosscheck
            rendering:
              show_root_heading: true
              show_root_toc_entry: false
              heading_level: 4

    The recipe returns a `handle` function which `bokeh` can use to visualize an annotation interface in multiple settings.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t1-1-active-learning.txt!}
</pre><br>

{!docs/snippets/html/juniper.html!}
