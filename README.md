![Hover](https://raw.githubusercontent.com/phurwicz/hover/main/docs/images/hover-logo-title.png)

> Explore and label on a map of raw data.
>
> Get enough to feed your model in no time.

[![PyPI Version](https://img.shields.io/pypi/v/hover?logo=pypi&logoColor=white)](https://pypi.org/project/hover/)
[![Downloads](https://static.pepy.tech/personalized-badge/hover?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/hover)
[![Build Status](https://img.shields.io/github/workflow/status/phurwicz/hover/python-package?logo=github&logoColor=white)](https://github.com/phurwicz/hover/actions)
[![Codacy Grade](https://img.shields.io/codacy/grade/689827d9077b43ac8721c7658d122d1a?logo=codacy&logoColor=white)](https://app.codacy.com/gh/phurwicz/hover/dashboard)
[![Codacy Coverage](https://img.shields.io/codacy/coverage/689827d9077b43ac8721c7658d122d1a/main?logo=codacy&logoColor=white)](https://app.codacy.com/gh/phurwicz/hover/dashboard)

`hover` speeds up data labeling through `embedding + visualization + callbacks`.
-   all you need to supply is raw data and a vectorizer function.

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

-   [Example script with walkthrough](https://phurwicz.github.io/hover/pages/tutorial/t0-quickstart/), ready to run in the browser
-   [Example ![annotation interface](https://img.shields.io/badge/annotation-interface-579ACA.svg?logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-linked-annotator) hosted on Binder

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
