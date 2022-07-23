"""
Convert tutorial scripts to Jupyter notebooks.
"""
import os
import re
import nbformat
import subprocess
from local_helper import batch_routine
from local_config import (
    NAME_TO_SCRIPT_ABS,
    THEBE_PATTERN_WITH_TAGS,
    THEBE_PATTERN_CODE_ONLY,
)

SNIPPETS_HOME = os.path.join(os.path.dirname(__file__), "../../")

SNIPPETS_PATTERN = r"\{\!([^\n\{\!\}]+?)\!\}"

SNIPPETS_TO_IGNORE = {
    "docs/snippets/html/thebe.html",
    "docs/snippets/markdown/binder-kernel.md",
    # "docs/snippets/markdown/local-dependency.md",
    # "docs/snippets/markdown/local-dep-text.md",
    # "docs/snippets/markdown/local-dep-jupyter-bokeh.md",
}

SYNTAX_CAPTURE_TO_REPLACE = {
    # markdown-material example: `??? info "FAQ"` block
    # multi-line expandable
    r"(?<=\n)(\?{3}[\s\S]+?)(?=(?:\n\S|$))": [
        # most outer ??? block
        (
            r'(?<=^)\?{3}[\s\S][^\n]+?"([^\n"]+?)"([\s\S]+?)(?=(?:\n\S|$))',
            r"-   <details open><summary>\1</summary>\2\n</details>\n\n",
        ),
        # 1st nested ??? block
        (
            r'(?<=\n)\s{4}\?{3}[\s\S][^\n]+?"([^\n"]+?)"([\s\S]+?)(?=(?:\n\s{0,4}\S|$))',
            r"    <details open><summary>\1</summary>\2\n    </details>\n",
        ),
        # 2nd nested ??? block
        (
            r'(?<=\n)\s{8}\?{3}[\s\S][^\n]+?"([^\n"]+?)"([\s\S]+?)(?=(?:\n\s{0,8}\S|$))',
            r"        <details open><summary>\1</summary>\2\n        </details>\n",
        ),
        # fences of code blocks need to remove indentation
        (r"(?<=\n)\s+```", r"```"),
    ],
}


def main():
    """
    Convert all mkdocs-material-thebe scripts to notebooks.
    """
    batch_routine(markdown_to_notebook, NAME_TO_SCRIPT_ABS)


def preprocess_markdown_include(markdown_content):
    """
    Subroutine for ignoring or pasting markdown-include snippets.
    """
    output = markdown_content
    pieces = re.split(SNIPPETS_PATTERN, output)

    # process markdown-include
    eff_pieces = []
    for i, _p in enumerate(pieces):
        # case 1: not markdown-include, keep as-is
        if i % 2 == 0:
            eff_pieces.append(_p)
            continue

        # case 2: markdown-include but the snippet is to be ignored
        if _p in SNIPPETS_TO_IGNORE:
            continue

        # case 3: markdown-include, replace snippet path by its content
        _path = os.path.join(SNIPPETS_HOME, _p)
        with open(_path, "r") as f:
            eff_pieces.append(f"\n{f.read()}\n")

    output = "".join(eff_pieces)
    return output


def preprocess_mkdocs_material(markdown_content):
    """
    Subroutine for converting mkdocs-material syntax into simple markdown.
    """
    output = markdown_content

    # process mkdocs-material
    for _capture_pattern, _replace_pairs in SYNTAX_CAPTURE_TO_REPLACE.items():
        pieces = re.split(_capture_pattern, output)
        eff_pieces = []
        for i, _p in enumerate(pieces):
            # even indices do not come from capture group
            if i % 2 == 0:
                eff_pieces.append(_p)
                continue

            # odd indices come from capture group
            for _from, _to in _replace_pairs:
                _p = re.sub(_from, _to, _p)
            eff_pieces.append(_p)

        output = "".join(eff_pieces)
    return output


def preprocess_markdown(markdown_content):
    """
    Aggregate procedure for turning multi-plugin markdown into simple markdown.
    """
    content = preprocess_markdown_include(markdown_content)
    content = preprocess_mkdocs_material(content)
    return content


def postprocess_snippet(text):
    """
    Clean a snippet that is about to be put in a cell.
    """
    text = re.sub(r"^\n+", "", text)
    text = re.sub(r"\n\n+", "\n\n", text)
    text = re.sub(r"\n+$", "", text)
    return text


def markdown_to_notebook(script_name, source_abs_path):
    """
    Turn a mkdocs-material markdown file into a notebook.
    """
    notebook_path = f"{script_name}.ipynb"

    with open(source_abs_path, "r") as f_source:
        source = preprocess_markdown(f_source.read())
        markdown_pieces = re.split(THEBE_PATTERN_WITH_TAGS, source)
        script_pieces = re.findall(THEBE_PATTERN_CODE_ONLY, source)
        script = "\n".join(script_pieces)

    assert (
        len(markdown_pieces) == len(script_pieces) + 1
    ), "Expected exactly one more markdown piece than script"

    cells = []
    while script_pieces:
        text = postprocess_snippet(markdown_pieces.pop(0))
        code = postprocess_snippet(script_pieces.pop(0))
        cells.append(nbformat.v4.new_markdown_cell(text))
        cells.append(nbformat.v4.new_code_cell(code))

    while markdown_pieces:
        text = postprocess_snippet(markdown_pieces.pop(0))
        cells.append(nbformat.v4.new_markdown_cell(text))

    nb = nbformat.v4.new_notebook()
    nb["cells"] = cells
    nbformat.write(nb, notebook_path)

    process = subprocess.run(
        ["jupyter", "nbconvert", "--execute", "--inplace", notebook_path],
        capture_output=True,
        timeout=1200,
    )
    return script, process


if __name__ == "__main__":
    main()
