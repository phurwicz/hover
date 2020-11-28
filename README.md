# Hover

[![PyPI Stage](https://img.shields.io/pypi/status/hover?style=for-the-badge)](https://pypi.org)
[![PyPI Version](https://img.shields.io/pypi/v/hover?style=for-the-badge)](https://pypi.org)
[![Travis CI](https://img.shields.io/travis/com/phurwicz/hover/main?style=for-the-badge)](https://travis-ci.com)
[![Codacy Grade](https://img.shields.io/codacy/grade/689827d9077b43ac8721c7658d122d1a?style=for-the-badge)](https://www.codacy.com)
[![Codacy Coverage](https://img.shields.io/codacy/coverage/689827d9077b43ac8721c7658d122d1a/main?style=for-the-badge)](https://www.codacy.com)

[![Demo](docs/images/app-linked-annotator.gif)

----

`Hover` is a **machine teaching** library that enables intuitive and effecient supervision. In other words, it provides a map where you _hover_ over and label your data... differently. For instance, you can:

-   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-simple-annotator) :seedling: annotate an intuitively selected group of data points at a time
-   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-snorkel-annotator) :whale: cross-check with Snorkel-based distant supervision
-   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-active-annotator) :ferris_wheel: **UPCOMING** throw a model in the loop and take advantage of active learning

Check out [@phurwicz/hover-binder](https://github.com/phurwicz/hover-binder) for a complete list of demo apps.

## Features

Here we attempt a quick comparison with a few other packages that do machine teaching:

Package        | `Hover`                               | `Prodigy`                               | `Snorkel`
-------------- | ------------------------------------- | --------------------------------------- | -------------------------
Core idea      | supervise like painting a picture     | scriptable active learning              | programmatic distant supervision
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

## Installation

To get the latest release version, you can use `pip`:

```bash
pip install hover
```

Installation through `conda` is not yet supported.

## Resources

-   [Documentation](https://phurwicz.github.io/hover/)

## Dependencies

-   `./requirements.txt` lists the dependencies for installation.
-   `./requirements-test.txt` lists additional dependencies for the test suite.
-   `./requirements-dev.txt` lists the packages for developers.
