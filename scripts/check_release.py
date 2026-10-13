"""Fail closed on invalid releases or distributions containing changed SDK source."""

import argparse
from email.parser import BytesParser
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tarfile
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile


def git_bytes(*args: str) -> bytes:
    """Read committed state without depending on working-tree file contents."""
    return subprocess.check_output(["git", *args])


def validate_tag(tag: str, version: str) -> None:
    """Require a stable, matching vMAJOR.MINOR.PATCH release tag."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version) or tag != f"v{version}":
        raise ValueError(f"Release tag {tag!r} must match stable version v{version}")


def verify_metadata(data: bytes, version: str) -> None:
    """Require the public package identity and intended version."""
    metadata = BytesParser().parsebytes(data)
    if metadata["Name"] != "qpiai-quantum" or metadata["Version"] != version:
        raise ValueError("Distribution name/version does not match public project")


def verify_sources(files: dict[str, bytes], expected: dict[str, bytes]) -> None:
    """Reject missing, changed, extra, or internal package source files."""
    package = {
        name: data for name, data in files.items() if name.startswith("qpiai_quantum/")
    }
    missing = {name for name in expected if name.endswith(".py")} - package.keys()
    if missing:
        raise ValueError(f"Missing package source: {sorted(missing)}")
    for name, data in package.items():
        if "_internal" in Path(name).parts or name.endswith(".joblib"):
            raise ValueError(f"Forbidden public artifact content: {name}")
        if name not in expected or expected[name] != data:
            raise ValueError(f"Artifact differs from committed source: {name}")


def verify_artifacts(directory: Path, version: str) -> None:
    """Check exactly one wheel and sdist, including every packaged SDK file."""
    expected = {
        name: git_bytes("show", f"HEAD:{name}")
        for name in git_bytes("ls-files", "qpiai_quantum").decode().splitlines()
    }
    wheel = directory / f"qpiai_quantum-{version}-py3-none-any.whl"
    sdist = directory / f"qpiai_quantum-{version}.tar.gz"
    if set(directory.iterdir()) - {directory / ".gitignore"} != {wheel, sdist}:
        raise ValueError("Expected exactly the intended public wheel and sdist")
    with zipfile.ZipFile(wheel) as archive:
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
    verify_sources(files, expected)
    verify_metadata(files[f"qpiai_quantum-{version}.dist-info/METADATA"], version)
    with tarfile.open(sdist) as source_archive:
        source_files = {}
        prefix = f"qpiai_quantum-{version}/"
        for member in source_archive.getmembers():
            if member.isfile():
                handle = source_archive.extractfile(member)
                assert handle is not None
                if not member.name.startswith(prefix):
                    raise ValueError("Unexpected sdist root")
                source_files[member.name.removeprefix(prefix)] = handle.read()
    verify_sources(source_files, expected)
    verify_metadata(source_files["PKG-INFO"], version)
    for artifact in (wheel, sdist):
        print(
            f"{artifact.name}: sha256={hashlib.sha256(artifact.read_bytes()).hexdigest()}"
        )


def main() -> None:
    """Validate a release event or the artifacts produced from its commit."""
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("source", "artifacts"))
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    version = tomllib.loads(git_bytes("show", "HEAD:pyproject.toml").decode())[
        "project"
    ]["version"]
    if args.mode == "artifacts":
        verify_artifacts(args.dist, version)
        return
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"], check=True
    )
    if os.environ["EVENT_NAME"] == "release":
        tag = os.environ["RELEASE_TAG"]
        validate_tag(tag, version)
        if git_bytes("rev-parse", f"refs/tags/{tag}^{{commit}}") != git_bytes(
            "rev-parse", "HEAD"
        ):
            raise ValueError("Checkout does not match the release tag")
        try:
            with urlopen(
                f"https://pypi.org/pypi/qpiai-quantum/{version}/json", timeout=30
            ):
                pass
        except HTTPError as error:
            if error.code != 404:
                raise
        else:
            raise ValueError(f"Version {version} already exists on PyPI")
    elif os.environ["EVENT_NAME"] != "workflow_dispatch":
        raise ValueError("Unsupported workflow event")
    print(
        f"Validated public source at {git_bytes('rev-parse', 'HEAD').decode().strip()}: {version}"
    )


if __name__ == "__main__":
    main()
