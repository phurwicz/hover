import os
from markdown_include.include import MarkdownInclude


DIR_PATH = os.path.dirname(__file__)
NAME_TO_SCRIPT_REL = {
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
NAME_TO_SCRIPT_ABS = {
    _k: os.path.join(DIR_PATH, _v) for _k, _v in NAME_TO_SCRIPT_REL.items()
}


MARKDOWN_INCLUDE = MarkdownInclude(
    configs={
        "base_path": os.path.join(DIR_PATH, "../../"),
        "encoding": "utf-8",
    }
)

THEBE_PATTERN_CODE_ONLY = r"(?<=<pre data-executable>)[\s\S]*?(?=</pre>)"
THEBE_PATTERN_WITH_TAGS = r"<pre data-executable>[\s\S]*?</pre>"
