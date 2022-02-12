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

<pre data-executable>
{!docs/snippets/py/t7-0a-lf-list-edit.txt!}
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

Unlike the toggled filters for `finder` and `softlabel`, filtering with functions is on a per-click basis. In other words, this particular filtration doesn't persist when you select another area.

## **Dynamic List of Functions**

Python lists are mutable, and we are going to take advantage of that for improvising and editing labeling functions on the fly.

Run the block below and open the resulting URL to launch a recipe.

-   labeling functions are evaluated against the `dev` set.
    -   hence you are advised to send the labels produced by these functions to the `train` set, not the `dev` set.
-   come back and edit the list of labeling functions **in-place** in one of the code cells above.
    -   then go to the launched app and refresh the functions!

<pre data-executable>
{!docs/snippets/py/t7-3-snorkel-crosscheck.txt!}

{!docs/snippets/py/tz-bokeh-server-notebook.txt!}
</pre>

What's really cool is that in your local environment, this update-and-refresh operation can be done all in a notebook. So now you can

-   interactively evaluate and revise labeling functions
-   visually assign specific data regions to apply those functions

which is, as this tutorial claims, *the* two things that significantly boost the accuracy of labeling functions.

{!docs/snippets/html/stylesheet.html!}
