"""Validate and publish this checkout's wheel and source archive to PyPI.

Use --configure-token to enter a token locally and store it in the system keyring.
Use --check for validation without uploading. Build with `python -m build` first.
"""

import argparse
import configparser
from email.parser import BytesParser
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile

import keyring
from packaging.utils import canonicalize_name
from packaging.version import Version

UPLOAD_URL = "https://upload.pypi.org/legacy/"
ROOT = Path(__file__).resolve().parents[1]


def configure_token():
    token = getpass.getpass("PyPI API token (input hidden): ").strip()
    if not token.startswith("pypi-"):
        raise ValueError("Expected a PyPI API token starting with pypi-.")
    keyring.set_password(UPLOAD_URL, "__token__", token)
    print("PyPI token saved in the system credential manager.")


def has_token():
    if os.environ.get("TWINE_PASSWORD"):
        return True
    if keyring.get_password(UPLOAD_URL, "__token__"):
        return True
    configuration = configparser.ConfigParser(interpolation=None)
    configuration.read(Path.home() / ".pypirc", encoding="utf-8")
    return bool(configuration.get("pypi", "password", fallback=""))


def validate_artifacts(root=ROOT):
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    name, version = project["name"], project["version"]
    prefix = canonicalize_name(name).replace("-", "_")
    artifacts = [
        root / "dist" / f"{prefix}-{version}-py3-none-any.whl",
        root / "dist" / f"{prefix}-{version}.tar.gz",
    ]
    for artifact in artifacts:
        if not artifact.is_file():
            raise ValueError(f"Missing {artifact.name}; run 'python -m build' first.")
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                metadata = [entry for entry in archive.namelist() if entry.endswith(".dist-info/METADATA")]
                if len(metadata) != 1:
                    raise ValueError(f"Expected one METADATA file in {artifact.name}.")
                content = archive.read(metadata[0])
        else:
            with tarfile.open(artifact) as archive:
                metadata = [entry for entry in archive.getmembers() if len(Path(entry.name).parts) == 2 and entry.name.endswith("/PKG-INFO")]
                if len(metadata) != 1:
                    raise ValueError(f"Expected one root PKG-INFO file in {artifact.name}.")
                content = archive.extractfile(metadata[0]).read()
        headers = BytesParser().parsebytes(content)
        if canonicalize_name(headers["Name"]) != canonicalize_name(name) or Version(headers["Version"]) != Version(version):
            raise ValueError(f"Metadata does not match pyproject.toml: {artifact.name}")
    return name, version, artifacts


def check_remote(name, version):
    try:
        with urlopen(f"https://pypi.org/pypi/{canonicalize_name(name)}/json", timeout=25) as response:
            project = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            print(f"No existing PyPI project found for {name}; this will be its first release.")
            return
        raise
    if any(Version(existing) == Version(version) for existing in project.get("releases", {})):
        raise ValueError(f"{name} {version} already exists on PyPI. Published versions cannot be replaced.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--configure-token", action="store_true")
    mode.add_argument("--check", action="store_true", help="Validate artifacts and release availability without uploading")
    args = parser.parse_args(argv)
    try:
        if args.configure_token:
            configure_token()
            return 0
        name, version, artifacts = validate_artifacts()
        subprocess.run([sys.executable, "-m", "twine", "check", "--strict", *map(str, artifacts)], check=True)
        check_remote(name, version)
        if not has_token():
            print("No PyPI token configured. Run: python scripts/publish_pypi.py --configure-token")
            return 0 if args.check else 1
        if args.check:
            print(f"{name} {version}: artifacts and publishing credentials are ready; nothing uploaded.")
            return 0
        subprocess.run([
            sys.executable, "-m", "twine", "upload", "--non-interactive", "--disable-progress-bar",
            "--repository-url", UPLOAD_URL, "--username", "__token__", *map(str, artifacts),
        ], check=True)
        print(f"Published: https://pypi.org/project/{canonicalize_name(name)}/{version}/")
    except (ValueError, OSError, subprocess.CalledProcessError, keyring.errors.KeyringError) as error:
        print(f"Publication stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
