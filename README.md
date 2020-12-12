![Hover](docs/images/hover-logo-title.png)

> Imagine editing a picture layer by layer, not pixel by pixel, nor by splashing paint.
>
> We can apply this idea to datasets.

[![PyPI Stage](https://img.shields.io/pypi/status/hover?style=for-the-badge)](https://pypi.org)
[![PyPI Version](https://img.shields.io/pypi/v/hover?style=for-the-badge)](https://pypi.org)
[![Build Workflow](https://img.shields.io/github/workflow/status/phurwicz/hover/python-package?style=for-the-badge)](https://github.com/features/actions)
[![Codacy Grade](https://img.shields.io/codacy/grade/689827d9077b43ac8721c7658d122d1a?style=for-the-badge)](https://www.codacy.com)
[![Codacy Coverage](https://img.shields.io/codacy/coverage/689827d9077b43ac8721c7658d122d1a/main?style=for-the-badge)](https://www.codacy.com)

![Demo](docs/images/app-active-learning.gif)

----

`Hover` is a **machine teaching** library that enables intuitive and effecient supervision. In other words, it provides a map where you _hover_ over and label your data... differently. For instance, you can:

-   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-simple-annotator) :seedling: annotate an intuitively selected group of data points at a time
-   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-active-learning) :ferris_wheel: throw a model in the loop and exploit active learning
-   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-snorkel-annotator) :whale: cross-check with Snorkel-based distant supervision

Check out [@phurwicz/hover-binder](https://github.com/phurwicz/hover-binder) for a complete list of demo apps.

## :flags: Latest Announcements

-   **Dec 12** Common usage workflows are now available in `hover.recipes`. Much cleaner code!
-   **Dec 6** Video tutorials coming soon! The first one is projected to be available by Dec 15.

## :flight_departure: Quick Start

`Hover` uses [`bokeh`](https://bokeh.org) to build its annotation interface:

```python
# app-annotator.py

from hover.core.explorer import BokehCorpusAnnotator
from bokeh.io import curdoc

# df is a pandas dataframe with 2D embedding
# which hover can help you compute

annotator = BokehCorpusAnnotator({"raw": df})
annotator.plot()

curdoc().add_root(annotator.view())
curdoc().title = "Simple-Annotator"
```

```bash
bokeh serve app-annotator.py
```

The most exciting features of `Hover` employ lots of Python callbacks, for which [`bokeh serve`](https://docs.bokeh.org/en/latest/docs/user_guide/server.html) comes into play.

## :package: Installation

> Python: 3.6+
>
> OS: tested on Mac & Linux

To get the latest release version, you can use `pip`:

```bash
pip install hover
```

Installation through `conda` is not yet available. Please open an issue if you would like `conda` or `conda-forge` support.

## :flamingo: Features

Here we attempt a quick comparison with a few other packages that do machine teaching:

Package        | `Hover`                               | [`Prodigy`](https://prodi.gy)           | [`Snorkel`](https://snorkel.ai)
-------------- | ------------------------------------- | --------------------------------------- | -------------------------
Core idea      | supervise like editing a picture      | scriptable active learning              | programmatic distant supervision
Annotates per  | batch of just the size you find right | piece predicted to be the most valuable | the whole dataset as long as it fits in
Supports       | all classification (text only atm)    | text & images, audio, vidio, & more     | text classification (for the most part)
Status         | open-source                           | proprietary                             | open-source
Devs           | indie                                 | Explosion AI                            | Stanford / Snorkel AI
Related        | many imports of the awesome `Bokeh`   | builds on the `Thinc`/`SpaCy` stack     | Variants: `Snorkel Drybell`, `MeTaL`, `DeepDive`
Vanilla usage  | define a vectorizer and annotate away | choose a base model and annotate away   | define labeling functions and apply away
Advanced usage | combine w/ active learning & snorkel  | patterns / transformers / custom models | transforming / slicing functions
Hardcore usage | exploit `hover.core` templates        | custom @prodigy.recipe                  | the upcoming `Snorkel Flow`

`Hover` claims the best deal of scale vs. precision thanks to

-   the flexibility to use, or not use, any technique beyond annotating on a "map";
-   the speed, or coarseness, of annotation being _literally at your fingertips_;
-   the interaction between multiple "maps" that each serves a different but connected purpose.

## :book: Resources

### [Documentation](https://phurwicz.github.io/hover/)

-   Contains API references of the most crucial components.
-   A lot more is on the way! (video tutorials, for example)

### Dependencies

-   `./requirements-test.txt` lists additional dependencies for the test suite.
-   `./requirements-dev.txt` lists recommended packages for developers.
