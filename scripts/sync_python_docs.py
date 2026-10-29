"""Synchronize this repository's Python tutorials into a YiGraphDocs checkout."""

import argparse
from pathlib import Path
import shutil

PAGES = [
    Path("docs-site/docs/tutorial-basics/python-package.md"),
    Path("docs-site/i18n/en/docusaurus-plugin-content-docs/current/tutorial-basics/python-package.md"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docs_repository", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.docs_repository.resolve()
    if not (target / "docs-site" / "docusaurus.config.js").is_file():
        parser.error("Expected a YiGraphDocs checkout containing docs-site/docusaurus.config.js")
    for page in PAGES:
        destination = target / page
        if args.check:
            if not destination.is_file() or destination.read_bytes() != (root / page).read_bytes():
                parser.exit(1, f"Out of sync: {destination}\n")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / page, destination)
        print(destination)


if __name__ == "__main__":
    main()
