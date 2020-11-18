import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fd:
    deps = [_line for _line in fd.read().split("\n") if not _line.startswith("#")]

setuptools.setup(
    name="hover",
    version="0.1.3",
    description="Hovercraft-like machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pavel",
    author_email="pepsimixt@gmail.com",
    url="https://github.com/phurwicz/hover",
    packages=setuptools.find_packages(),
    install_requires=deps,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
