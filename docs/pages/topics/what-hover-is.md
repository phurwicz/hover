## :sparkles: Features

Here we attempt a quick comparison with a few other packages that do machine teaching:

Package        | **Hover**                             | [**Prodigy**](https://prodi.gy)         | [**Snorkel**](https://snorkel.ai)
-------------- | ------------------------------------- | --------------------------------------- | -------------------------
Core idea      | batch annotation with extensions      | scriptable active learning              | programmatic distant supervision
Annotates per  | batch of just the size you find right | piece predicted to be the most valuable | the whole dataset as long as it fits in
Supports       | all classification (text only atm)    | text & images, audio, vidio, & more     | text classification (for the most part)
Status         | open-source                           | proprietary                             | open-source
Devs           | indie                                 | Explosion AI                            | Stanford / Snorkel AI
Related        | many imports of the awesome `Bokeh`   | builds on the `Thinc`/`SpaCy` stack     | variants: `Snorkel Drybell`, `MeTaL`, `DeepDive`
Vanilla usage  | define a vectorizer and annotate away | choose a base model and annotate away   | define labeling functions and apply away
Advanced usage | combine w/ active learning & snorkel  | patterns / transformers / custom models | transforming / slicing functions
Hardcore usage | exploit `hover.core` templates        | custom @prodigy.recipe                  | the upcoming `Snorkel Flow`

`Hover` claims the best deal of scale vs. precision thanks to

-   the flexibility to use, or not use, any technique beyond annotating on a "map";
-   the speed, or coarseness, of annotation being _literally at your fingertips_;
-   the interaction between multiple "maps" that each serves a different but connected purpose.
