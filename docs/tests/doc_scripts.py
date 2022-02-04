"""
Tests the scripts in the docs.
"""
import os
import uuid

PY_SNIPPET_HOME = "docs/snippets/py"
SCRIPT_TO_SNIPPETS = {
    "tutorial-t0": [
        "t0-0-dataset-text.txt",
        "t0-1-vectorizer.txt",
        "t0-2-reduction.txt",
        "t0-3-simple-annotator.txt",
    ],
    "tutorial-t1": [
        "t0-0-dataset-text.txt",
        "t0-1-vectorizer.txt",
        "t0-2-reduction.txt",
        "t1-0-vecnet-callback.txt",
        "t1-1-active-learning.txt",
    ],
    # t2 has no script currently
    "tutorial-t3": [
        "t0-0-dataset-text.txt",
        "t3-0-dataset-population-table.txt",
        "t3-1-dataset-commit-dedup.txt",
        "t3-2-dataset-selection-table.txt",
        "t3-3-dataset-evict-patch.txt",
    ],
}


def main():
    for _script_name, _snippet_paths in SCRIPT_TO_SNIPPETS.items():
        print(f"======== Running {_script_name} ========")
        create_script_and_run(_script_name, _snippet_paths)


def create_script_and_run(script_name, snippet_paths):
    script_path = f"{script_name}-{uuid.uuid1()}.py"
    with open(script_path, "w") as f_script:
        for _path in snippet_paths:
            with open(os.path.join(PY_SNIPPET_HOME, _path), "r") as f_snippet:
                f_script.write(f_snippet.read())
                f_script.write("\n")

    os.system(f"python {script_path}")


if __name__ == "__main__":
    main()
