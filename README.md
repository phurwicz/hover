![Hover](https://raw.githubusercontent.com/phurwicz/hover/main/docs/images/hover-logo-title.png)

> Explore and label on a map of raw data.
>
> Get enough to feed your model in no time.

[![PyPI Version](https://img.shields.io/pypi/v/hover?logo=pypi&logoColor=white)](https://pypi.org/project/hover/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/hover)](https://github.com/conda-forge/hover-feedstock)
[![Downloads](https://static.pepy.tech/personalized-badge/hover?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=pypi%20downloads)](https://pepy.tech/project/hover)
[![Build Status](https://img.shields.io/github/workflow/status/phurwicz/hover/python-package?logo=github&logoColor=white)](https://github.com/phurwicz/hover/actions)
[![Codacy Grade](https://img.shields.io/codacy/grade/689827d9077b43ac8721c7658d122d1a?logo=codacy&logoColor=white)](https://app.codacy.com/gh/phurwicz/hover/dashboard)
[![Codacy Coverage](https://img.shields.io/codacy/coverage/689827d9077b43ac8721c7658d122d1a/main?logo=codacy&logoColor=white)](https://app.codacy.com/gh/phurwicz/hover/dashboard)

`hover` speeds up data labeling through `embedding + visualization + callbacks`.

-   You just need raw data and an embedding to start.

![Demo](https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.5.0/trailer.gif)

## :sparkles: Features

> **It's fast because it labels in bulk.**

:telescope: A 2D-embedded view of your dataset for labeling, equipped with

-   **Tooltip** for each point and **table view** for groups of points.
-   **Search** widgets for ad-hoc highlight of data matching search criteria.
-   **Toggle** buttons that clearly distinguish data subsets ("raw"/"train"/"dev"/"test").

> **It's accurate because you can filter and extend.**

:microscope: Supplementary views to provide further labeling precision, such as

-   Advanced search view which can **filter points by search criteria** and provides stronger highlight.
-   Active learning view which puts a model in the loop and can **filter by confidence score**.
-   Function-based view which can leverage **custom functions for labeling and filtering**.

> **It's fun because the process never gets old.**

-   Explore the map to find out which "zones" are easy and which ones are tricky.
-   Join the conquest of your data by coloring all of those zones through wisdom!

Check out [@phurwicz/hover-binder](https://github.com/phurwicz/hover-binder) for a list of demo apps.

## :rocket: Quickstart

### [**Code + Walkthrough -> Labeling App**](https://phurwicz.github.io/hover/pages/tutorial/t0-quickstart/)

-   edit & run code right in your browser, with guides along the way.

### [**Jump to Labeling App**](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-linked-annotator)

-   interactive plot for labeling data, pre-built and hosted on Binder.

## :package: Install

> Python: 3.7+
>
> OS: Linux & Mac & Windows

PyPI (for all releases): `pip install hover`

Conda-forge (for 0.6.0 and above): `conda install -c conda-forge hover`

For Windows users, we recommend [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about).

-   On Windows itself you will need [C++ build tools](https://visualstudio.microsoft.com/downloads/) for dependencies.

## :book: Resources

-   [Binder repo](https://github.com/phurwicz/hover-binder)
-   [Changelog](https://github.com/phurwicz/hover/blob/main/CHANGELOG.md)
-   [Documentation](https://phurwicz.github.io/hover/)
-   [Tutorials](https://phurwicz.github.io/hover/pages/tutorial/t0-quickstart/)

## :flags: Project News

-   **Feb 25, 2022** version 0.7.0 is now available. Check out the [changelog](https://github.com/phurwicz/hover/blob/main/CHANGELOG.md) for details :partying_face:. Some tl-dr for the impatient:
    -   **audio and image support** supply audio/image files through URLs to label with `hover`!
        -   any type supported by HTML (and your browser) will be supported here.
    -   **high-dimensional support** you can now use higher-than-2D embeddings.
        -   `hover` still plots in 2D, but you can dynamically choose which two dimension to use.

## :bell: Remarks

### Shoutouts

-   Thanks to [`Bokeh`](https://bokeh.org) because `hover` would not exist without linked plots and callbacks, or be nearly as good without embeddable server apps.
-   Thanks to [Philip Vollet](https://de.linkedin.com/in/philipvollet) for sharing `hover` with the community even when it was really green.

### Contributing

-   All feedbacks are welcome, **especially what you find lacking and want it fixed!**
-   `./requirements-dev.txt` lists required packages for development.
-   Pull requests are advised to use a superset of the pre-commit hooks listed in [.pre-commit-config.yaml](https://github.com/phurwicz/hover/blob/main/.pre-commit-config.yaml).

### Citation

If you have found `hover` useful to your work, please [let us know](https://github.com/phurwicz/hover/discussions) :hugs:

```tex
@misc{hover,
  title={{hover}: label data at scale},
  url={https://github.com/phurwicz/hover},
  note={Open software from https://github.com/phurwicz/hover},
  author={
    Pavel Hurwicz and
    Haochuan Wei},
  year={2021},
}
```
