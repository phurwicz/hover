> The most common usage of `hover` is through built-in "recipes" like in the quickstart.
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

> In general, a "recipe" is a function which takes a `SupervisableDataset` and other arguments based on what the recipe does.
>
> -   Note how `active_learning` takes more arguments than `simple_annotator` due to, well, active learning.
>
> The recipe returns a "handle" function which `bokeh` can use to visualize an annotation interface in multiple settings.

???+ tip "Basic tips"
    Inspecting model predictions allows us to

    -   get an idea of how the current set of annotations will likely teach the model.
    -   locate the most valuable samples for further annotation.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t1-1-active-learning.txt!}
</pre><br>

{!docs/snippets/html/juniper.html!}
