# CI Tagged Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `.github/workflows/ci.yml` so that pushing a `v*` git tag runs the full pipeline and publishes a versioned Docker image to GHCR.

**Architecture:** Three small edits to the existing `ci.yml`: add a tag push trigger, loosen the deploy job condition to include tag refs, and add `type=semver` tag patterns to `docker/metadata-action`. No new files.

**Tech Stack:** GitHub Actions, `docker/metadata-action@v5`, `docker/build-push-action@v6`

---

### Task 1: Add tag trigger and update deploy condition

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add `tags: ['v*']` to the push trigger**

In `.github/workflows/ci.yml`, replace the `on` block:

```yaml
on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
```

- [ ] **Step 2: Update the deploy job condition**

In the `deploy` job, replace:

```yaml
    if: github.ref == 'refs/heads/main'
```

with:

```yaml
    if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/')
```

- [ ] **Step 3: Validate the YAML is well-formed**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "OK"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run deploy job on tag pushes"
```

---

### Task 2: Add semver image tags to metadata-action

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Replace the metadata-action `tags` block**

In the `deploy` job, find the `Extract metadata` step and replace its `tags` input:

```yaml
          tags: |
            type=sha
            type=raw,value=latest
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
```

The full step should read:

```yaml
      - name: Extract metadata (tags, labels) for Docker
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha
            type=raw,value=latest
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
```

- [ ] **Step 2: Validate the YAML is well-formed**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "OK"
```

Expected: `OK`

- [ ] **Step 3: Verify final state of the deploy job**

```bash
grep -A 40 "name: Push container image" .github/workflows/ci.yml
```

Expected output should show:
- `if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/')`
- `type=semver,pattern={{version}}` in the tags block
- `type=semver,pattern={{major}}.{{minor}}` in the tags block

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add semver image tags for tagged releases"
```
