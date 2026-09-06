# Publishing PhyloGeoPlot to PyPI

This document describes how releases of `phylogeoplot` are built and published,
both automatically (via GitHub Actions and PyPI Trusted Publishing) and manually.

## Overview

- CI (`.github/workflows/ci.yml`) runs on every push/PR to `main`: it installs the
  package with its `dev` extras, runs the test suite, and builds the sdist/wheel
  with `python -m build` to catch packaging regressions early.
- Publishing (`.github/workflows/publish.yml`) builds the distribution and uploads
  it using [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
  (OIDC) — no API tokens are stored in the repository.
  - `workflow_dispatch` with `target: testpypi` publishes to **TestPyPI**.
  - `workflow_dispatch` with `target: pypi` **or** a published GitHub **release**
    publishes to **PyPI**.

## 1. Version bump / tag flow

1. Update the `version` field in [`pyproject.toml`](../pyproject.toml) following
   [Semantic Versioning](https://semver.org/) (e.g. `0.1.4` -> `0.1.5`).
2. Commit the version bump (e.g. `chore: bump version to 0.1.5`) and merge it to
   `main` via a pull request.
3. Create an annotated git tag matching the version, prefixed with `v`:

   ```bash
   git tag -a v0.1.5 -m "Release 0.1.5"
   git push origin v0.1.5
   ```

## 2. GitHub release flow

1. On GitHub, go to **Releases -> Draft a new release**.
2. Choose the tag created above (or create it from the UI).
3. Fill in release notes (highlights, breaking changes, contributors).
4. Click **Publish release**.
5. Publishing the release triggers the `release: published` event, which runs
   the `publish-pypi` job in `.github/workflows/publish.yml` and uploads the
   build to **PyPI** automatically.

## 3. Required PyPI Trusted Publisher configuration

Trusted Publishing must be configured once per package on both PyPI and
TestPyPI before the workflow can upload without a token.

On [pypi.org](https://pypi.org/manage/account/publishing/) and
[test.pypi.org](https://test.pypi.org/manage/account/publishing/), add a new
pending/trusted publisher with:

| Field                  | Value                              |
| ---------------------- | ----------------------------------- |
| PyPI Project Name      | `phylogeoplot`                     |
| Owner                  | `tahiri-lab`                       |
| Repository name        | `PhyloGeoPlot`                     |
| Workflow name          | `publish.yml`                      |
| Environment name       | `pypi` (PyPI) / `testpypi` (TestPyPI) |

The workflow declares matching GitHub Actions **environments** (`pypi` and
`testpypi`) with `permissions: id-token: write`, which is what allows the OIDC
exchange to succeed. If the project does not exist yet on PyPI/TestPyPI, use a
*pending* trusted publisher, which will attach itself to the project on first
successful publish.

## 4. Testing a release with TestPyPI

Before cutting a real release, verify the packaging on TestPyPI:

1. Go to **Actions -> Publish to PyPI -> Run workflow**.
2. Select `target: testpypi` and run it on the branch/tag you want to test.
3. Once it succeeds, install from TestPyPI in a clean virtual environment:

   ```bash
   python -m venv .venv-testpypi
   source .venv-testpypi/bin/activate
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ phylogeoplot
   ```

4. Confirm the package imports and behaves as expected before publishing to
   the real PyPI index.

## 5. Manual publishing (without GitHub Actions)

If you need to publish locally:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*   # TestPyPI dry run
python -m twine upload dist/*                          # PyPI
```

`twine` will prompt for credentials; use a scoped PyPI API token
(`__token__` as the username) if Trusted Publishing is not available in your
environment.

## 6. Rollback / fix-forward guidance

PyPI does not allow re-uploading a file for a version that has already been
published or deleted, so releases should be treated as immutable:

- **Do not** attempt to delete and re-upload the same version number.
- If a release is broken, publish a new patch version (e.g. `0.1.5` ->
  `0.1.6`) with the fix, following the flow above.
- You may **yank** a broken release on PyPI (Project -> Release -> "Yank
  release") to discourage new installs while keeping it available for
  projects already pinned to that version.
- Communicate the issue and the fixed version in the GitHub release notes of
  the follow-up release.
