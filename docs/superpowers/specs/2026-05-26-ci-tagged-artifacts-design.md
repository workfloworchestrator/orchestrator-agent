# CI Tagged Artifacts Design

## Goal

Extend the existing `ci.yml` workflow so that pushing a `v*` git tag triggers the full test → build → deploy pipeline and publishes a versioned Docker image to GHCR alongside the existing `latest`/`sha` tags.

## Scope

Single file change: `.github/workflows/ci.yml`. No new workflow files.

## Changes

### 1. Trigger

Add `tags: ['v*']` to the `on.push` block so the test matrix and downstream jobs run on tag pushes.

```yaml
on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
```

### 2. Deploy condition

Allow the deploy job to fire on both `main` and any tag ref:

```yaml
if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/')
```

### 3. Docker metadata tags

Add `type=semver` patterns so tagged releases produce version-specific image tags:

```yaml
tags: |
  type=sha
  type=raw,value=latest
  type=semver,pattern={{version}}
  type=semver,pattern={{major}}.{{minor}}
```

Resulting tags per event:

| Event | Image tags |
|-------|-----------|
| Push to `main` | `sha-<hash>`, `latest` |
| Push tag `v1.2.3` | `sha-<hash>`, `latest`, `1.2.3`, `1.2` |

## Non-goals

- Python wheel / PyPI publishing (not in scope)
- Tag validation or release notes automation
- Changing the test matrix or lint workflow
