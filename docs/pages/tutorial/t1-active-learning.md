> `hover` offers more powerful built-in recipes.
>
> :ferris_wheel: Let's explore an active learning example.

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
</pre><br>


## **Ingredient 4 / 4: Model Callback**

To utilize active learning, we need to specify how to get a model in the loop:

??? info "Custom models"
    It is possible to use a model other than `VectorNet` or its subclass.

    Simply implement the following methods with the same signatures as `VectorNet`:

    -   [`train`](../../reference/core-neural/#hover.core.neural.VectorNet.train)
    -   [`save`](../../reference/core-neural/#hover.core.neural.VectorNet.save)
    -   [`predict_proba`](../../reference/core-neural/#hover.core.neural.VectorNet.predict_proba)

<pre data-executable>
{!docs/snippets/py/t1-0-vecnet-callback.txt!}
</pre><br>


## :sparkles: **Recipe time**

Now we invoke the `active_learning` recipe.

???+ tip "Basic tips"
    Inspecting model predictions allows us to

    -   get an idea of how the current set of annotations will likely teach the model.
    -   locate the most valuable samples for further annotation.

{!docs/snippets/markdown/jupyterlab-js-issue.md!}

<pre data-executable>
{!docs/snippets/py/t1-1-active-learning.txt!}
</pre><br>

{!docs/snippets/html/juniper.html!}
