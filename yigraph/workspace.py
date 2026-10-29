"""Explicit workspace creation; importing YiGraph never writes user files."""

from importlib.resources import files
from pathlib import Path
import shutil


def init_workspace(path):
    root = Path(path).expanduser().resolve()
    package = files("aag")
    templates = files("yigraph").joinpath("resources")
    copies = [
        (templates.joinpath("engine_config.yaml"), root / "config" / "engine_config.yaml"),
        (templates.joinpath("data_upload_config.yaml"), root / "config" / "data_upload_config.yaml"),
        (package.joinpath("computing_engine", "config_servers.yaml"), root / "config" / "config_servers.yaml"),
    ]

    def copy_tree(source, destination):
        for child in source.iterdir():
            target = destination / child.name
            if child.is_dir():
                copy_tree(child, target)
            elif child.name != "__init__.py" and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with child.open("rb") as stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)

    for source, destination in copies:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as stream, destination.open("xb") as output:
                shutil.copyfileobj(stream, output)
    # Only data resources are copied, never the knowledge management Python code.
    for source in package.joinpath("knowledge_base").iterdir():
        if source.name.endswith(".yaml"):
            destination = root / "knowledge_base" / source.name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as stream, destination.open("xb") as output:
                    shutil.copyfileobj(stream, output)
    copy_tree(package.joinpath("datasets"), root / "datasets")
    return root
