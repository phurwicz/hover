> `hover` manages data through a `SupervisableDataset` class.
>
> Here we walk through some basic behaviors and interactions that can turn out useful.

{!docs/pages/tutorial/snippet-stylesheet.html!}

<pre data-executable>
from hover.core.dataset import SupervisableTextDataset

# ---- simplistic data for illustation ----
my_data = {
    "raw": [
        {"text": "Avocados are my favorite!"},
        {"text": "Blueberries are not bad either."},
        {"text": "Citrus ... sure why not"},
    ],
    "train": [
        {"text": "Citrus ... sure why not", "label": "C"},
        {"text": "Dragonfruits cost too much", "label": "D"},
    ],
    "dev": [
        {"text": "Dragonfruits cost too much", "label": "D"},
        {"text": "Eggplants? Not in this scope.", "label": "E"},
    ],
    "test": [
        {"text": "Eggplants? Not in this scope.", "label": "E"},
    ]

}
# -----------------------------------

dataset = SupervisableTextDataset(
    raw_dictl=my_data["raw"],
    train_dictl=my_data["train"],
    dev_dictl=my_data["dev"],
    test_dictl=my_data["test"],
    # "text" is the default feature field for SupervisableTextDataset
    # "label" is the default label field for SupervisableDataset
)

# Be aware of the automatic deduplication by feature
# which keeps test > dev > train > raw
dataset.dfs
</pre><br>

{!docs/pages/tutorial/snippet-juniper.html!}
