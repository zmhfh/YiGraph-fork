"""Ensure release checks cannot accidentally upload an existing version."""

from io import BytesIO
import json
from pathlib import Path

import pytest

from scripts import publish_pypi


@pytest.mark.parametrize("check_only", [False, True])
def test_missing_credentials_never_upload(monkeypatch, check_only):
    calls = []
    monkeypatch.setattr(publish_pypi, "validate_artifacts", lambda: ("yigraph", "2.0.0", [Path("release.whl")]))
    monkeypatch.setattr(publish_pypi, "check_remote", lambda *_: None)
    monkeypatch.setattr(publish_pypi, "has_token", lambda: False)
    monkeypatch.setattr(publish_pypi.subprocess, "run", lambda arguments, **_: calls.append(arguments))
    assert publish_pypi.main(["--check"] if check_only else []) == (0 if check_only else 1)
    assert not any("upload" in call for call in calls)


def test_check_mode_never_uploads_with_valid_credentials(monkeypatch):
    calls = []
    monkeypatch.setattr(publish_pypi, "validate_artifacts", lambda: ("yigraph", "2.0.0", [Path("release.whl")]))
    monkeypatch.setattr(publish_pypi, "check_remote", lambda *_: None)
    monkeypatch.setattr(publish_pypi, "has_token", lambda: True)
    monkeypatch.setattr(publish_pypi.subprocess, "run", lambda arguments, **_: calls.append(arguments))
    assert publish_pypi.main(["--check"]) == 0
    assert not any("upload" in call for call in calls)


def test_equivalent_published_version_is_rejected(monkeypatch):
    monkeypatch.setattr(publish_pypi, "urlopen", lambda *_args, **_kwargs: BytesIO(json.dumps({"releases": {"2.0": []}}).encode()))
    with pytest.raises(ValueError, match="already exists"):
        publish_pypi.check_remote("yigraph", "2.0.0")
