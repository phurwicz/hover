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
    version="0.2.4",
    description="Data annotation done right: easy, fun, hyper-productive, and inducing insight.",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    author="Pavel",
    author_email="pavelhurwicz@gmail.com",
    url="https://github.com/phurwicz/hover",
    packages=setuptools.find_packages(),
    install_requires=[
        # interactive/static visualization
        "bokeh>=2.2.3",
        # data handling
        "pandas>=1.1.4",
        "numpy>=1.14",
        "scipy>=1.3.2",
        "numba>=0.46.0",
        "scikit-learn>=0.21.0",
        "umap-learn>=0.3.10",
        "ivis[cpu]>=1.7",
        # distant supervision
        "snorkel>=0.9.6",
        # neural stuff
        "torch>=1.4.0",
        # utilities
        "tqdm>=4.0",
        "wasabi>=0.4.0",
        "wrappy>=0.2.6",
        "rich>=9.2.0",
    ],
    python_requires=">=3.6, <3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
