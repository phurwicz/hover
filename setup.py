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
    version="0.3.2",
    description="Data annotation done right: easy, fun, hyper-productive, and inducing insight.",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    author="Pavel",
    author_email="pavelhurwicz@gmail.com",
    url="https://github.com/phurwicz/hover",
    packages=setuptools.find_packages(),
    install_requires=[
        # interactive/static visualization
        "bokeh",
        # distant supervision
        "snorkel>=0.9.3",
        "scikit-learn",
        # neural stuff
        "torch>=1.4.0",
        # data handling
        "pandas>=1.1.4",
        "numpy>=1.14",
        # computations
        "scipy>=1.3.2",
        # utilities
        "tqdm>=4.0",
        "wasabi>=0.4.0",
        "wrappy>=0.2.6",
        "rich>=9.2.0",
        # optional: dimensionality reduction
        # "umap-learn>=0.3.10",
        # "ivis[cpu]>=1.7",
    ],
    python_requires=">=3.6, <3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
