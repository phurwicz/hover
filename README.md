![Hover](https://raw.githubusercontent.com/phurwicz/hover/main/docs/images/hover-logo-title.png)

> Explore and label on a map of your data.
>
> Get enough to feed your model in no time.

[![en](https://img.shields.io/badge/lang-en-green.svg)](https://github.com/phurwicz/hover/blob/main/README.md)
[![zh](https://img.shields.io/badge/语言-中文-green.svg)](https://github.com/phurwicz/hover/blob/main/README.zh.md)

[![PyPI Version](https://img.shields.io/pypi/v/hover?logo=pypi&logoColor=white)](https://pypi.org/project/hover/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/hover)](https://github.com/conda-forge/hover-feedstock)
![Downloads](https://static.pepy.tech/personalized-badge/hover?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=pypi%20downloads)
![Main Build Status](https://img.shields.io/github/actions/workflow/status/phurwicz/hover/cross-os-source-test.yml?branch=main&label=main&logo=github)
![Nightly Build Status](https://img.shields.io/github/actions/workflow/status/phurwicz/hover/quick-source-test.yml?branch=nightly&label=nightly&logo=github)
![Codacy Grade](https://img.shields.io/codacy/grade/689827d9077b43ac8721c7658d122d1a?logo=codacy&logoColor=white)
![Codacy Coverage](https://img.shields.io/codacy/coverage/689827d9077b43ac8721c7658d122d1a/main?logo=codacy&logoColor=white)

`hover` is a tool for mass-labeling data points that can be represented by vectors.

-   Labeling is as easy as coloring a scatter plot.
-   Hover your mouse and lasso-select to inspect any cluster.
-   Use a variety of widgets to narrow down further.
-   Enter a suitable label and hit "Apply"!

![GIF Demo](https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.5.0/trailer-short.gif)

## :rocket: Live Demos

### [**With code**](https://phurwicz.github.io/hover/pages/tutorial/t0-quickstart/)

-   edit & run code in your browser to get a labeling interface, with guides along the way.

### [**Without code**](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-simple-annotator)

-   go directly to an example labeling interface hosted on Binder.

## :sparkles: Features

> **It's fast because it labels data in bulk.**

:telescope: A semantic scatter plot of your data for labeling, equipped with

<details open>
  <summary> <b>Tooltip</b> for each point on mouse hover </summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/image-tooltip.gif">
</details>

<details>
  <summary> Table view for <b>inspecting selected</b> points </summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/selection-table.gif">
</details>

<details>
  <summary> Toggle buttons that clearly <b>distinguish data subsets</b> </summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/subset-toggle.gif">
</details>

<details>
  <summary> <b>Search</b> widgets for ad-hoc data highlight </summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/text-search-response.gif">
</details>

> **It's accurate because multiple components work together.**

:microscope: Supplementary views to use in conjunction with the annotator, including

<details>
  <summary> `Finder`: <b>filter</b> data by search criteria</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/finder-filter.gif">
</details>

<details>
  <summary> `SoftLabel`: <b>active learning</b> by in-the-loop model prediction score</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/active-learning.gif">
</details>

<details>
  <summary> `Snorkel`: <b>custom functions</b> for labeling and filtering</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/labeling-function.gif">
</details>

> **It's flexible (and fun!) because the process never gets old.**

:toolbox: Additional tools and options that allow you to

<details>
  <summary> Go to <b>higher dimensions</b> (3D? 4D?) and choose your xy-axes </summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/change-axes.gif">
</details>

<details>
  <summary> <b>Consecutively select</b> across areas, dimensions, and views</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/keep-selecting.gif">
</details>

<details>
  <summary> <b>Kick outliers</b> and <b>fix mistakes</b></summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/evict-and-patch.gif">
</details>

## :package: Install

> Python: 3.8+
>
> OS: Linux & Mac & Windows

PyPI: `pip install hover`

Conda: `conda install -c conda-forge hover`

## :book: Resources

-   [Tutorials](https://phurwicz.github.io/hover/pages/tutorial/t0-quickstart/)
-   [Binder repo](https://github.com/phurwicz/hover-binder)
-   [Changelog](https://github.com/phurwicz/hover/blob/main/CHANGELOG.md)
-   [Documentation](https://phurwicz.github.io/hover/)

## :flags: Announcements

-   **Jan 21, 2023** version 0.8.0 is now available. Check out the [changelog](https://github.com/phurwicz/hover/blob/main/CHANGELOG.md) for details :partying_face:.

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
