![Hover](https://raw.githubusercontent.com/phurwicz/hover/main/docs/images/hover-logo-title.png)

[![en](https://img.shields.io/badge/lang-en-green.svg)](https://github.com/phurwicz/hover/blob/main/README.md)

> 通过向量降维, 极速探索和批量标注数据, 并用作模型训练或其它用途.

[![PyPI Version](https://img.shields.io/pypi/v/hover?logo=pypi&logoColor=white)](https://pypi.org/project/hover/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/hover)](https://github.com/conda-forge/hover-feedstock)
![Downloads](https://static.pepy.tech/personalized-badge/hover?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=pypi%20downloads)
![Main Build Status](https://img.shields.io/github/actions/workflow/status/phurwicz/hover/cross-os-source-test.yml?branch=main&label=main&logo=github)
![Nightly Build Status](https://img.shields.io/github/actions/workflow/status/phurwicz/hover/quick-source-test.yml?branch=nightly&label=nightly&logo=github)
![Codacy Grade](https://img.shields.io/codacy/grade/689827d9077b43ac8721c7658d122d1a?logo=codacy&logoColor=white)
![Codacy Coverage](https://img.shields.io/codacy/coverage/689827d9077b43ac8721c7658d122d1a/main?logo=codacy&logoColor=white)

`hover` 是一个批量标注数据的工具, 只需数据能被向量表示.

-   标注过程很简单, 如同给散点图上色.
-   通过移动鼠标和框选, 来观察数据(在降维后的)点簇.
-   使用小工具(如搜索/过滤/规则/主动学习)来提升精度.
-   输入合适的标签, 并点击"Apply"按钮, 即可标注!

![GIF Demo](https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.5.0/trailer-short.gif)

## :rocket: 在线演示

### [**With code**](https://phurwicz.github.io/hover/pages/tutorial/t0-quickstart/)

-   查看含代码的教程, 可在浏览器中编辑和运行, 无需安装依赖.

### [**Without code**](https://mybinder.org/v2/gh/phurwicz/hover-binder/master?urlpath=/proxy/5006/app-simple-annotator)

-   跳过所有代码, 进入托管在Binder上的标注界面.

## :sparkles: 具体功能

:telescope: 将向量降维得到二维数据散点图, 并配有

<details open>
  <summary> <b>提示框</b> 来显示具体数据内容 </summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/image-tooltip.gif">
</details>

<details>
  <summary> 表格来 <b>批量检视</b> 选中的数据 </summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/selection-table.gif">
</details>

<details>
  <summary> 切换按钮来 <b>区分数据子集</b> </summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/subset-toggle.gif">
</details>

<details>
  <summary> <b>文本/正则匹配</b> 来定向搜寻数据 </summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/text-search-response.gif">
</details>

:microscope: 与标注界面同步的辅助模式

<details>
  <summary> `Finder`: 以匹配条件来 <b>过滤</b> 选中的数据</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/finder-filter.gif">
</details>

<details>
  <summary> `SoftLabel`: <b>主动学习</b> 用模型打分过滤选中的数据</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/active-learning.gif">
</details>

<details>
  <summary> `Snorkel`: <b>自定义函数</b> 来过滤数据或直接打标</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/labeling-function.gif">
</details>

:toolbox: 更多的补充工具

<details>
  <summary> 降维时保留 <b>更多维度</b> (3D? 4D?) 并动态选择观察的平面</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/change-axes.gif">
</details>

<details>
  <summary> 跨界面/跨维度地进行 <b>持续选取/反选</b> 以达到更高精度</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/keep-selecting.gif">
</details>

<details>
  <summary> <b>剔除</b>选中数据中的异类 以及 <b>修订</b>发现的误标</summary>
  <img src="https://raw.githubusercontent.com/phurwicz/hover-gallery/main/0.7.0/evict-and-patch.gif">
</details>

## :package: 安装

> Python: 3.8+
>
> 操作系统: Linux & Mac & Windows

PyPI: `pip install hover`

Conda-forge: `conda install -c conda-forge hover`

## :book: 资料

-   [教程](https://phurwicz.github.io/hover/pages/tutorial/t0-quickstart/)
-   [Binder仓库](https://github.com/phurwicz/hover-binder)
-   [版本说明](https://github.com/phurwicz/hover/blob/main/CHANGELOG.md)
-   [文档](https://phurwicz.github.io/hover/)

## :flags: 新动态

-   **Jan 21, 2023** version 0.8.0 is now available. Check out the [changelog](https://github.com/phurwicz/hover/blob/main/CHANGELOG.md) for details :partying_face:.

## :bell: 其它说明

### 鸣谢和推荐

-   我们推荐 [`Bokeh`](https://bokeh.org) 可视化框架, `hover`正是基于它的图表同步和回调函数来实现非常重要的功能.
-   感谢 [Philip Vollet](https://de.linkedin.com/in/philipvollet) 在`hover`的迭代早期 无偿地帮助在开源社区内推广.

### 提供贡献

-   我们欢迎任何反馈, **特别是使用中的痛点!**
-   `./requirements-dev.txt` 列出了开发者所需的依赖.
-   我们建议在提交PR前启用[.pre-commit-config.yaml](https://github.com/phurwicz/hover/blob/main/.pre-commit-config.yaml)中列出的pre-commit hook.

### Citation

如果`hover`对您的工作有帮助, 请[告诉我们](https://github.com/phurwicz/hover/discussions)或引用 :hugs:

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
