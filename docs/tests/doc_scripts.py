"""
Tests the scripts in the docs.
"""
import os
import re
import uuid
import markdown
import subprocess
from rich.console import Console
from markdown_include.include import MarkdownInclude


DIR_PATH = os.path.dirname(__file__)
NAME_TO_SCRIPT = {
    "tutorial-t0": "../pages/tutorial/t0-quickstart.md",
    "tutorial-t1": "../pages/tutorial/t1-active-learning.md",
    # tutorial-t2 has no script currently
    "tutorial-t3": "../pages/tutorial/t3-dataset-population-selection.md",
    "tutorial-t4": "../pages/tutorial/t4-annotator-dataset-interaction.md",
    "tutorial-t5": "../pages/tutorial/t5-finder-filter.md",
    "tutorial-t6": "../pages/tutorial/t6-softlabel-joint-filter.md",
    "tutorial-t7": "../pages/tutorial/t7-snorkel-improvise-rules.md",
    "tutorial-t8": "../pages/tutorial/t8-recipe-structure-customization.md",
}


MARKDOWN_INCLUDE = MarkdownInclude(
    configs={
        "base_path": os.path.join(DIR_PATH, "../../"),
        "encoding": "utf-8",
    }
)

THEBE_PATTERN = r"(?<=<pre data-executable>)[\s\S]*?(?=</pre>)"

CONSOLE = Console()


def main():
    """
    Test all code blocks in the scripts listed in this file.
    Collect all exceptions along the way.
    """
    exceptions = {}
    for _name, _path in NAME_TO_SCRIPT.items():
        CONSOLE.print(f"======== Running {_name} ========")
        _retval = parse_script_and_run(_name, _path)
        if isinstance(_retval, Exception):
            exceptions[_name] = _retval

    if exceptions:
        for _name, _exception in exceptions.items():
            CONSOLE.print(f"Caught error from {_name}: {_exception}")

        CONSOLE.print("Please check rich traceback above.")
        raise RuntimeError("Script test failed.")


def parse_script_and_run(script_name, source_rel_path):
    """
    Retrieve and run code blocks from documentation file.
    Note that the doc file can  using markdown-include.
    """
    source_abs_path = os.path.join(DIR_PATH, source_rel_path)
    script_tmp_path = f"{script_name}-{uuid.uuid1()}.py"

    with open(source_abs_path, "r") as f_source:
        source = f_source.read()
        html = markdown.markdown(source, extensions=[MARKDOWN_INCLUDE])
        script = "\n".join(re.findall(THEBE_PATTERN, html))

    with open(script_tmp_path, "w") as f_script:
        f_script.write(script)

    try:
        subprocess.run(["python", script_tmp_path], check=True)
        return None
    except Exception as e:
        CONSOLE.print(f"!!!!!!!! Error in {script_name} !!!!!!!!", style="red bold")
        CONSOLE.print(script, style="blue")
        CONSOLE.print_exception(show_locals=False)
        return e


if __name__ == "__main__":
    main()
