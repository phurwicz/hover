> Suppose we have some custom functions for labeling or filtering data, which resembles [`snorkel`](https://github.com/snorkel-team/snorkel)'s typical scenario.
>
> :speedboat: Let's see how these functions can be combined with `hover`.

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}
{!docs/snippets/markdown/component-tutorial.md!}

## **Preparation**

{!docs/snippets/markdown/dataset-prep.md!}

## **Labeling Functions**

Labeling functions are functions that **take a `pd.DataFrame` row and return a label or abstain**.

Inside the function one can do many things, but let's start with simple keywords wrapped in regex:

??? info "About the decorator @labeling_function"
    ::: hover.utils.snorkel_helper.labeling_function

<pre data-executable>
{!docs/snippets/py/t7-0-lf-list.txt!}
</pre><br>

### **Using a Function to Apply Labels**

Hover's `SnorkelExplorer` (short as `snorkel`) can take the labeling functions above and apply them on areas of data that you choose. The widget below is responsible for labeling:

<pre data-executable>
{!docs/snippets/py/tz-bokeh-notebook.txt!}

{!docs/snippets/py/t7-1-snorkel-apply-button.txt!}
</pre><br>

### **Using a Function to Apply Filters**

Any function that labels is also a function that filters. The filter condition is `"keep if did not abstain"`. The widget below handles filtering:

<pre data-executable>
{!docs/snippets/py/t7-2-snorkel-filter-button.txt!}
</pre><br>

#### **Unlike Filter Toggles, This is One-Time**

## **In Jupyter: Change Function List Dynamically**

Excellent for improvisation of new rules.


{!docs/snippets/html/stylesheet.html!}
