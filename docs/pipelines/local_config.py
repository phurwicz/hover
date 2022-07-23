import os
from markdown_include.include import MarkdownInclude


DIR_PATH = os.path.dirname(__file__)
NAME_TO_SCRIPT_REL = {
    "t0-quickstart": "../pages/tutorial/t0-quickstart.md",
    "t1-using-recipes": "../pages/tutorial/t1-active-learning.md",
    # tutorial-t2 has no script currently
    "t3-dataset-mechanisms": "../pages/tutorial/t3-dataset-population-selection.md",
    "t4-annotator-plot-tools": "../pages/tutorial/t4-annotator-dataset-interaction.md",
    "t5-finder-selection-filter": "../pages/tutorial/t5-finder-filter.md",
    "t6-soft-label-joint-filters": "../pages/tutorial/t6-softlabel-joint-filter.md",
    "t7-custom-labeling-functions": "../pages/tutorial/t7-snorkel-improvise-rules.md",
    "g0-image-data": "../pages/guides/g0-datatype-image.md",
    "g1-audio-data": "../pages/guides/g1-datatype-audio.md",
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
