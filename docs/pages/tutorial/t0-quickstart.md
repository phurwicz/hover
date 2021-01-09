> Welcome to the minimal guide of `hover`!
>
> :sunglasses: Let's label some data and call it a day.

{!docs/pages/tutorial/snippet-stylesheet.html!}

## **Ingredient 1 / 3: Some Data**

Suppose that we have a list of data entries, each in the form of a dictionary:

<pre data-executable>
from hover.core.dataset import SupervisableTextDataset
from faker import Faker
import random

# ---- fake data for illustation ----
fake_en = Faker("en")

def random_text():
    return fake_en.paragraph(3)

def random_raw_data():
    return {"content": random_text()}

def random_labeled_data():
    return {"content": random_text(), "mark": random.choice(["A", "B"])}
# -----------------------------------

dataset = SupervisableTextDataset(
    # raw data which do not have labels
    raw_dictl=[random_raw_data() for i in range(500)],
    # train / dev / test sets are optional
    train_dictl=[],
    dev_dictl=[random_labeled_data() for i in range(50)],
    test_dictl=[random_labeled_data() for i in range(50)],
    # adjust feature_key and label_key to your data
    feature_key="content",
    label_key="mark",
)

# each subset is stored in its own DataFrame
dataset.dfs["raw"].head(5)
</pre><br>


## **Ingredient 2 / 3: Vectorizer**

To put our dataset sensibly on a 2-D "map", we will use a vectorizer for feature extraction, and then perform dimensionality reduction.<br>

Here's one way to define a vectorizer:

<pre data-executable>
import spacy
import re

nlp = spacy.load("en_core_web_md")

def vectorizer(text):
    clean_text = re.sub(r"[\s]+", r" ", text)
    return nlp(clean_text, disable=nlp.pipe_names).vector

text = dataset.dfs["raw"].loc[0, "text"]
vec = vectorizer(text)
print(f"Text: {text}")
print(f"Vector shape: {vec.shape}")

</pre><br>


## **Ingredient 3 / 3: Reduction**

The dataset has built-in high-level support for dimensionality reduction. <br>
Currently we can use [umap](https://umap-learn.readthedocs.io/en/latest/) or [ivis](https://bering-ivis.readthedocs.io/en/latest/).

??? info "Optional dependencies"
    The corresponding libraries do not ship with hover by default, and may need to be installed:

    -   for umap: `pip install umap-learn`
    -   for ivis: `pip install ivis[cpu]` or `pip install ivis[gpu]`

    `umap-learn` is installed in this demo environment.

<pre data-executable>
# any kwargs will be passed onto the corresponding reduction
# for umap: https://umap-learn.readthedocs.io/en/latest/parameters.html
# for ivis: https://bering-ivis.readthedocs.io/en/latest/api.html
dataset.compute_2d_embedding(vectorizer, "umap")

# What we did adds 'x' and 'y' columns to the DataFrames in dataset.dfs
# One could alternatively pre-compute these columns using any approach
dataset.dfs["raw"].head(5)
</pre><br>


## :sparkles: **Apply Labels**

Now we are ready to visualize and annotate!

???+ tip "Basic tips"
    There should be a `SupervisableDataset` board on the left and an `BokehDataAnnotator` on the right.

    The `SupervisableDataset` comes with a few buttons:

    -   `push`: push `Dataset` updates to the bokeh plots.
    -   `commit`: add data entries selected in the `Annotator` to a specified subset.
    -   `dedup`: deduplicate across subsets (keep the last entry).

    The `BokehDataAnnotator` comes with a few buttons:

    -   `raw`/`train`/`dev`/`test`: choose which subsets to display.
    -   `apply`: apply the `label` input to the selected points in the `raw` subset only.
    -   `export`: save your data (all subsets) in a specified format.

??? info "Best practices"
    We've essentially put the data into neighborboods based on the vectorizer, but the quality, or the homogeneity of labels, of such neighborhoods can vary.

    -   hover over any data point to see its tooltip.
    -   take advantage of different selection tools to apply labels at appropriate scales.
    -   the search widget might turn out useful.
        -    note that it does not select points but highlights them.

<pre data-executable>
from hover.recipes import simple_annotator
from bokeh.io import show, output_notebook

# 'handle' is a function that renders elements in bokeh documents
handle = simple_annotator(dataset)

output_notebook()
show(handle)
</pre><br>

{!docs/pages/tutorial/snippet-juniper.html!}
