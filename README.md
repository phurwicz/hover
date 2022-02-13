![Hover](https://raw.githubusercontent.com/phurwicz/hover/main/docs/images/hover-logo-title.png)

> Explore and label on a map of raw data.
>
> Get enough to feed your model in no time.

[![PyPI Version](https://img.shields.io/pypi/v/hover?logo=pypi&logoColor=white)](https://pypi.org/project/hover/)
[![Downloads](https://static.pepy.tech/personalized-badge/hover?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/hover)
[![Build Status](https://img.shields.io/github/workflow/status/phurwicz/hover/python-package?logo=github&logoColor=white)](https://github.com/phurwicz/hover/actions)
[![Codacy Grade](https://img.shields.io/codacy/grade/689827d9077b43ac8721c7658d122d1a?logo=codacy&logoColor=white)](https://app.codacy.com/gh/phurwicz/hover/dashboard)
[![Codacy Coverage](https://img.shields.io/codacy/coverage/689827d9077b43ac8721c7658d122d1a/main?logo=codacy&logoColor=white)](https://app.codacy.com/gh/phurwicz/hover/dashboard)

`hover` speeds up data labeling through `embedding + visualization + callbacks`. You just need raw data and a vectorizer function to get started.

![Demo](https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.5.0/trailer.gif)

## :sparkles: Features

-   :telescope: A 2D-embedded view of your dataset for labeling, equipped with
    -   **Tooltip** for each point and **table view** for groups of points.
    -   **Search** widgets for ad-hoc highlight of data matching search criteria.
    -   **Toggle** buttons that clearly distinguish data subsets ("raw"/"train"/"dev"/"test").

-   :microscope: Supplementary views to provide further labeling precision, such as
    -   Advanced search view which can **filter points by search criteria** and provides stronger highlight.
    -   Active learning view which puts a model in the loop and can **filter by confidence score**.
    -   Function-based view which can leverage **custom functions for labeling and filtering**.

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

To get the latest release version: `pip install hover`

Starting on version 0.6.0, we are also on conda: `conda install -c conda-forge hover`

For Windows users, we recommend [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about).
-   On Windows itself you will need [C++ build tools](https://visualstudio.microsoft.com/downloads/) for dependencies like `umap-learn`.

## :book: Resources

-   [Binder repo](https://github.com/phurwicz/hover-binder)
-   [Changelog](https://github.com/phurwicz/hover/blob/main/CHANGELOG.md)
-   [Documentation](https://phurwicz.github.io/hover/)
-   [Tutorials](https://phurwicz.github.io/hover/pages/tutorial/t0-quickstart/)

## :flags: Project News

-   **Feb 12, 2022** version 0.6.0 is now available. Check out the [changelog](https://github.com/phurwicz/hover/blob/main/CHANGELOG.md) for details :partying_face:. Some tl-dr for the impatient:
    -   you can now edit selections, like kicking points from current selection or updating cells on the fly.
    -   you can now make cumulative selections.
    -   `SupervisableDataset` no longer maintains lists of dictionaries, but does everything through dataframes.
    -   `active_learning` now takes a `VectorNet` directly.
    -   `snorkel_crosscheck` allows you to label and filter through functions. You can change those functions dynamically without having to replot!

## :bell: Remarks

### Shoutouts

-   Thanks to [`Bokeh`](https://bokeh.org) because `hover` would not exist without linked plots and callbacks, or be nearly as good without embeddable server apps.
-   Thanks to [Philip Vollet](https://de.linkedin.com/in/philipvollet) for sharing `hover` with the community even when it was really green.

### Contributing

-   All feedbacks are welcome :hugs: Especially what you find frustrating and want fixed!
-   `./requirements-dev.txt` lists required packages for development.
-   Pull requests are advised to use a superset of the pre-commit hooks listed in [.pre-commit-config.yaml](https://github.com/phurwicz/hover/blob/main/.pre-commit-config.yaml).
