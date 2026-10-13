"""Regression coverage for release identity and artifact source verification."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).parents[1] / "scripts" / "check_release.py"
)
assert _spec is not None and _spec.loader is not None
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


@pytest.mark.parametrize("tag", ["v0.7.5", "0.7.4", "v0.7.4rc1", "v0.7.4/extra"])
def test_reject_mismatched_or_unstable_tag(tag: str) -> None:
    with pytest.raises(ValueError):
        release.validate_tag(tag, "0.7.4")


def test_accept_matching_stable_tag() -> None:
    release.validate_tag("v0.7.4", "0.7.4")


@pytest.mark.parametrize(
    "files",
    [
        {},
        {"qpiai_quantum/vqe.py": b"reverted"},
        {"qpiai_quantum/vqe.py": b"committed", "qpiai_quantum/stale.py": b"extra"},
        {
            "qpiai_quantum/vqe.py": b"committed",
            "qpiai_quantum/_internal/x.py": b"private",
        },
    ],
)
def test_reject_missing_modified_extra_or_internal_source(
    files: dict[str, bytes],
) -> None:
    with pytest.raises(ValueError):
        release.verify_sources(files, {"qpiai_quantum/vqe.py": b"committed"})


def test_accept_exact_source() -> None:
    files = {"qpiai_quantum/vqe.py": b"committed"}
    release.verify_sources(files, files)


@pytest.mark.parametrize(
    "name,version", [("private-sdk", "0.7.4"), ("qpiai-quantum", "0.7.5")]
)
def test_reject_wrong_distribution_identity(name: str, version: str) -> None:
    with pytest.raises(ValueError):
        release.verify_metadata(f"Name: {name}\nVersion: {version}\n".encode(), "0.7.4")
