# Publishing the public SDK

This process applies only to GitHub `qpiai/quantum-sdk` and PyPI `qpiai-quantum`.

## One-time configuration

Create the GitHub environment `pypi`, restricted to tags matching `v*`.
In the existing PyPI project's Publishing settings, add a GitHub Trusted Publisher:

- Owner: `qpiai`
- Repository: `quantum-sdk`
- Workflow filename: `release.yml`
- Environment: `pypi`

No PyPI password or API token belongs in GitHub secrets. The existing `API_KEY`
secret is solely for SDK tests. Only the publishing job receives `id-token: write`,
which lets GitHub prove the workflow identity to PyPI using OpenID Connect (OIDC).

## Validate without uploading

Run **Release to PyPI** manually on `main` from the Actions page or with:

```sh
gh workflow run release.yml --repo qpiai/quantum-sdk --ref main
```

This runs source checks, the existing Python CI matrix, a clean build, strict Twine
checks, and comparisons of packaged SDK files against committed source. The wheel
and source distribution are saved as workflow artifacts. The publishing job is
always skipped for manual runs, including when the current version exists on PyPI.

## Publish a future version

1. Update `project.version` in `pyproject.toml`, refresh `uv.lock` with `uv lock`,
   and commit the intended release source to `main`.
2. Create a matching version tag, such as `v0.7.5` for version `0.7.5`, on that commit.
3. Publish a stable GitHub Release for that tag.

Publishing the GitHub Release starts deployment automatically. The tag must match
the version and point to a commit reachable from `main`. Prereleases do not publish.
An existing PyPI version is rejected before CI or uploading. Successful CI is followed
by a clean build, then a separate job uploads the exact verified distributions using
Trusted Publishing. Never edit an existing published tag to release different source.

The manual validation run proves the build and checks, but cannot prove a real PyPI
upload. The first future release will exercise the complete authentication/upload path.
