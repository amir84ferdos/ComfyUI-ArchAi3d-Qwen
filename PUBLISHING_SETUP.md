# Publishing Setup Guide

This guide will help you set up automated publishing to both Comfy Registry and PyPI.

---

## ✅ Completed Setup

The following has already been configured:

- ✅ **pyproject.toml** - Complete package metadata for both Comfy Registry and PyPI
- ✅ **GitHub Actions Workflow** - `.github/workflows/publish.yml` for Comfy Registry
- ✅ **Git Tags** - v2.1.0 tag created and pushed
- ✅ **CHANGELOG** - Version history documentation
- ✅ **requirements.txt** - Dependencies file (PyYAML)

---

## 🔑 Step 1: Set Up Comfy Registry Publishing

### 1.1 Get Your Comfy Registry Access Token

1. **Visit the Comfy Registry**:
   - Go to: https://registry.comfy.org/

2. **Sign in or Create Account**:
   - Use your GitHub account to sign in

3. **Generate API Token**:
   - Navigate to your account settings
   - Look for "API Keys" or "Access Tokens" section
   - Click "Generate New Token"
   - Give it a descriptive name: "GitHub Actions - ComfyUI-ArchAi3d-Qwen"
   - Copy the token immediately (you won't see it again!)

### 1.2 Add Token to GitHub Repository Secrets

1. **Go to Your Repository Settings**:
   - Visit: https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/settings/secrets/actions

2. **Create New Repository Secret**:
   - Click "New repository secret"
   - **Name**: `REGISTRY_ACCESS_TOKEN` (exactly as shown - case sensitive!)
   - **Value**: Paste the token you copied from Comfy Registry
   - Click "Add secret"

### 1.3 Test Automated Publishing

The GitHub Action will now automatically run when:
- You push changes to `pyproject.toml` on the `main` branch
- You manually trigger it via "Actions" tab → "Publish to Comfy registry" → "Run workflow"

**To test immediately**:
1. Go to: https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/actions
2. Click "Publish to Comfy registry" workflow
3. Click "Run workflow" → Select "main" branch → "Run workflow"
4. Watch the workflow execute (should complete in 1-2 minutes)

---

## 📦 Step 2: Set Up PyPI Publishing (Optional)

### 2.1 Create PyPI Account

1. **Register on PyPI**:
   - Production: https://pypi.org/account/register/
   - Test (recommended first): https://test.pypi.org/account/register/

2. **Verify Your Email**:
   - Check your email and verify your account

### 2.2 Generate PyPI API Token

1. **Go to Account Settings**:
   - Production: https://pypi.org/manage/account/
   - Test: https://test.pypi.org/manage/account/

2. **Create API Token**:
   - Scroll to "API tokens" section
   - Click "Add API token"
   - **Token name**: "GitHub Actions - ComfyUI-ArchAi3d-Qwen"
   - **Scope**: Select "Entire account" (for first upload) or specific project (after first upload)
   - Click "Create token"
   - **IMPORTANT**: Copy the token starting with `pypi-` (you won't see it again!)

### 2.3 Add PyPI Token to GitHub Secrets

1. **Go to Repository Secrets**:
   - Visit: https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/settings/secrets/actions

2. **Create New Secret**:
   - Click "New repository secret"
   - **Name**: `PYPI_API_TOKEN` (exactly as shown!)
   - **Value**: Paste your PyPI token (starts with `pypi-`)
   - Click "Add secret"

3. **For Test PyPI** (optional, recommended for testing):
   - Create another secret named: `TEST_PYPI_API_TOKEN`
   - Use the token from test.pypi.org

### 2.4 Create PyPI Publishing Workflow

Create a new file: `.github/workflows/publish-pypi.yml`

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      test_pypi:
        description: 'Publish to Test PyPI instead of production'
        required: false
        default: 'false'

permissions:
  contents: read

jobs:
  publish-pypi:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    if: ${{ github.repository_owner == 'amir84ferdos' }}

    steps:
      - name: Check out code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to Test PyPI
        if: ${{ github.event.inputs.test_pypi == 'true' }}
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
        run: |
          twine upload --repository testpypi dist/*

      - name: Publish to PyPI
        if: ${{ github.event.inputs.test_pypi != 'true' }}
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          twine upload dist/*
```

### 2.5 Test PyPI Publishing

**Test on Test PyPI first** (recommended):
1. Go to Actions tab
2. Click "Publish to PyPI" workflow
3. Click "Run workflow"
4. Check "Publish to Test PyPI"
5. Click "Run workflow"
6. Verify at: https://test.pypi.org/project/comfyui-archai3d-qwen/

**Publish to Production PyPI**:
1. Create a GitHub Release:
   - Go to: https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/releases/new
   - Tag: v2.1.0 (select existing tag)
   - Title: "v2.1.0 - Automated Publishing & Bug Fixes"
   - Description: Copy from CHANGELOG.md
   - Click "Publish release"
2. The workflow will automatically run
3. Verify at: https://pypi.org/project/comfyui-archai3d-qwen/

---

## 🔄 How Automated Publishing Works

### Comfy Registry (Current Setup)

**Triggers**:
- ✅ Any push to `main` branch that modifies `pyproject.toml`
- ✅ Manual workflow dispatch

**What it does**:
1. Checks out your code
2. Publishes to Comfy Registry using the API token
3. Creates/updates the package listing
4. Users can now install via ComfyUI Manager

**Next publish will happen when**:
- You bump version in `pyproject.toml` (e.g., `version = "2.2.0"`)
- You push to main branch
- Workflow automatically publishes the new version

### PyPI (After Setup)

**Triggers**:
- ✅ GitHub Release is published
- ✅ Manual workflow dispatch

**What it does**:
1. Builds the Python package (wheel + source distribution)
2. Uploads to PyPI
3. Users can install with: `pip install comfyui-archai3d-qwen`

---

## 📝 Version Bumping Workflow

When you want to release a new version:

1. **Update version in `pyproject.toml`**:
   ```toml
   version = "2.2.0"  # Change this line
   ```

2. **Update `__init__.py`**:
   ```python
   __version__ = "2.2.0"
   ```

3. **Update CHANGELOG.md**:
   ```markdown
   ## [2.2.0] - 2025-XX-XX
   ### Added
   - New feature description
   ### Fixed
   - Bug fix description
   ```

4. **Commit and push**:
   ```bash
   git add pyproject.toml __init__.py CHANGELOG.md
   git commit -m "Bump version to 2.2.0"
   git push origin main
   ```

   ➡️ **Comfy Registry publishes automatically!**

5. **Create git tag**:
   ```bash
   git tag -a v2.2.0 -m "Release v2.2.0"
   git push origin v2.2.0
   ```

6. **Create GitHub Release** (for PyPI):
   - Go to: https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/releases/new
   - Select tag: v2.2.0
   - Title: "v2.2.0 - Brief Description"
   - Description: Copy from CHANGELOG.md
   - Click "Publish release"

   ➡️ **PyPI publishes automatically!**

---

## 🧪 Testing Your Setup

### Test Checklist

- [ ] **Comfy Registry Token Added**: Verify at Repository Settings → Secrets
- [ ] **Test Manual Publish**: Run workflow manually and check for success
- [ ] **Test Auto Publish**: Make a small change to `pyproject.toml` and push
- [ ] **Verify on Comfy Registry**: Search for your package
- [ ] **(Optional) PyPI Token Added**: If setting up PyPI
- [ ] **(Optional) Test PyPI Publish**: Run workflow with test mode
- [ ] **(Optional) Production PyPI Publish**: Create release and verify

### Troubleshooting

**Workflow fails with "Unauthorized"**:
- Check that secret name is exactly: `REGISTRY_ACCESS_TOKEN`
- Verify the token is valid and not expired
- Regenerate token if needed

**Workflow doesn't trigger on pyproject.toml changes**:
- Ensure you're pushing to `main` branch
- Check workflow file is in `.github/workflows/publish.yml`
- Verify workflow is enabled in Actions tab

**Package not appearing in Comfy Registry**:
- Wait 5-10 minutes for indexing
- Check workflow logs for errors
- Verify `pyproject.toml` has correct `[tool.comfy]` section

---

## 🎉 Success!

Once everything is set up, your publishing workflow will be:

1. **Develop features** → Commit code
2. **Ready to release** → Bump version in `pyproject.toml`
3. **Push to main** → Comfy Registry publishes automatically ✨
4. **Create GitHub Release** → PyPI publishes automatically ✨
5. **Users can install** → Via ComfyUI Manager or `pip install` 🚀

---

## 📞 Need Help?

If you encounter any issues:

1. Check the [GitHub Actions logs](https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/actions)
2. Review the [Comfy Registry documentation](https://registry.comfy.org/docs)
3. Read [PyPI publishing guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
4. Open an issue on GitHub

---

**Last Updated**: January 2025
**Version**: 2.1.0
