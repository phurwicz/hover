"""
Tests the scripts in the docs.
Intended for a Binder environment, and should be used in conjunction with a local libary in phurwicz/hover-binder.
"""
import re
import uuid
import markdown
import subprocess
from local_helper import batch_routine
from local_config import NAME_TO_SCRIPT_ABS, MARKDOWN_INCLUDE, THEBE_PATTERN_CODE_ONLY


def main():
    """
    Test all code blocks in the scripts listed in this file.
    Collect all exceptions along the way.
    """
    batch_routine(parse_script_and_run, NAME_TO_SCRIPT_ABS)


def parse_script_and_run(script_name, source_abs_path):
    """
    Retrieve and run code blocks from documentation file.
    Note that the doc file can be using markdown-include.
    """
    script_tmp_path = f"{script_name}-{uuid.uuid1()}.py"

    with open(source_abs_path, "r") as f_source:
        source = f_source.read()
        html = markdown.markdown(source, extensions=[MARKDOWN_INCLUDE])
        script = "\n".join(re.findall(THEBE_PATTERN_CODE_ONLY, html))

    with open(script_tmp_path, "w") as f_script:
        f_script.write(script)

    process = subprocess.run(
        ["python", script_tmp_path], capture_output=True, timeout=1200
    )
    return script, process


if __name__ == "__main__":
    main()
