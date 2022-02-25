"""
Tests the scripts in the docs.
Intended for a Binder environment, and should be used in conjunction with a local libary in phurwicz/hover-binder.
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
    "guide-g0": "../pages/guides/g0-datatype-image.md",
    "guide-g1": "../pages/guides/g1-datatype-audio.md",
}


MARKDOWN_INCLUDE = MarkdownInclude(
    configs={
        "base_path": os.path.join(DIR_PATH, "../../"),
        "encoding": "utf-8",
    }
)

THEBE_PATTERN = r"(?<=<pre data-executable>)[\s\S]*?(?=</pre>)"


def main():
    """
    Test all code blocks in the scripts listed in this file.
    Collect all exceptions along the way.
    """
    all_success = True
    console = Console()

    for _name, _path in NAME_TO_SCRIPT.items():
        console.print(f"======== Running {_name} ========")
        _script, _process = parse_script_and_run(_name, _path)
        _success = _process.returncode == 0
        all_success = all_success and _success

        if not _success:
            console.print(f"!!!!!!!! Error from {_name} !!!!!!!!", style="red bold")
            console.print(f"{_script}\n\n", style="blue")
            console.print(f"{_process.stderr}\n\n", style="red")

    if not all_success:
        raise RuntimeError("Script test failed.")


def parse_script_and_run(script_name, source_rel_path):
    """
    Retrieve and run code blocks from documentation file.
    Note that the doc file can be using markdown-include.
    """
    source_abs_path = os.path.join(DIR_PATH, source_rel_path)
    script_tmp_path = f"{script_name}-{uuid.uuid1()}.py"

    with open(source_abs_path, "r") as f_source:
        source = f_source.read()
        html = markdown.markdown(source, extensions=[MARKDOWN_INCLUDE])
        script = "\n".join(re.findall(THEBE_PATTERN, html))

    with open(script_tmp_path, "w") as f_script:
        f_script.write(script)

    process = subprocess.run(
        ["python", script_tmp_path], capture_output=True, timeout=1200
    )
    return script, process


if __name__ == "__main__":
    main()
