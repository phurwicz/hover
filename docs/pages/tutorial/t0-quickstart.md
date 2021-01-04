> Welcome to the minimal guide of `hover`!
>
> :sunglasses: Let's label some data and call it a day.

{!docs/pages/tutorial/snippet-stylesheet.html!}

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


The dataset has built-in support for dimensionality reduction. <br>
Currently we can use "umap" or "ivis"; however, the corresponding libraries do not ship with hover by default, and we may need to install them.

<pre data-executable>
# any kwargs will be passed onto the corresponding method
dataset.compute_2d_embedding(vectorizer, "umap")

# What we did adds 'x' and 'y' columns to the DataFrames in dataset.dfs
# One could alternatively pre-compute these columns using any method
dataset.dfs["raw"].head(5)
</pre><br>


Now we are ready to visualize and annotate!

<pre data-executable>
from hover.recipes import simple_annotator
from bokeh.io import show, output_notebook

# 'handle' is a function that renders elements in bokeh documents
handle = simple_annotator(dataset)

output_notebook()
show(handle)
</pre><br>

{!docs/pages/tutorial/snippet-juniper.html!}
