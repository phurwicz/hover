import os
from markdown_include.include import MarkdownInclude, IncludePreprocessor

README_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "README.md.template")
LANGUAGE_PLACEHOLDER = "<lang>"
LANGS = ["en", "zh"]
DEFAULT_LANG = "en"


def main():
    with open(README_TEMPLATE_PATH, "r") as f:
        template = f.read()
    include = MarkdownInclude()
    preprocessor = IncludePreprocessor(template, include.getConfigs())

    for lang in LANGS:
        filename = "README.md" if lang == DEFAULT_LANG else f"README.{lang}.md"
        readme_path = os.path.join(os.path.dirname(__file__), filename)
        transformed = "\n".join(
            preprocessor.run(template.replace(LANGUAGE_PLACEHOLDER, lang).split("\n"))
        )
        with open(readme_path, "w") as f:
            f.write(transformed)
            print(f"Generated {readme_path} for language {lang}.")


if __name__ == "__main__":
    main()
