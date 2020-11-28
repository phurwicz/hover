import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fd:
    deps = [_line for _line in fd.read().split("\n") if not _line.startswith("#")]

setuptools.setup(
    name="hover",
    version="0.2.2",
    description="Data annotation done right: easy, fun, hyper-productive, and inducing insight.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pavel",
    author_email="pavelhurwicz@gmail.com",
    url="https://github.com/phurwicz/hover",
    packages=setuptools.find_packages(),
    install_requires=deps,
    python_requires=">=3.6, <3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
