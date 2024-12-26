import setuptools
import os


def get_description():
    if os.path.isfile("README.md"):
        with open("README.md", "r") as fh:
            desc = fh.read()
    else:
        desc = ""
    return desc


setuptools.setup(
    name="hover",
    version="0.10.0",
    description="Label data at scale. Fun and precision included.",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    author="Pavel",
    author_email="pavelhurwicz@gmail.com",
    url="https://github.com/phurwicz/hover",
    packages=setuptools.find_packages(include=["hover*"]),
    install_requires=[
        # python-version-specific example: "numpy>=1.14,<=1.21.5;python_version<'3.8.0'",
        # interactive/static visualization
        "bokeh>=3.4.3",
        # preprocessors
        "scikit-learn>=1.4.0",
        # neural stuff
        "torch>=2.0.0",
        # data handling
        "pandas>=2.0.0",
        "polars>=1.10.0",
        "pyarrow>=18.0.0",
        # "numpy>=2.0.0",
        "numpy>=1.25.0,<2.0.0",
        # computations
        "scipy>=1.10.0",
        # utilities
        "tqdm>=4.0",
        "rich>=11.0.0",
        "deprecated>=1.1.0",
        # dimensionality reduction: UMAP is included
        "umap-learn>=0.5.0",
        # module config customization
        "flexmod>=0.1.2",
        # optional: more dimensionality reduction methods
        # "ivis[cpu]>=1.7",
        # optional: distant supervision
        # "snorkel>=0.9.8",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
