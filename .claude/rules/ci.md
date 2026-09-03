# CI Pipeline

Key workflows in `.github/workflows/` (non-exhaustive):

- **ci.yml**: Test matrix (ubuntu/macos/windows), ruff lint+format, mypy, Trivy filesystem scan, Hadolint, dependency-review (PRs only), Docker build + Trivy image scan
- **pr-title.yml**: Conventional commit PR title validation (`feat:`, `fix:`, `chore:`, etc.)
- **docker.yml**: Publish to ghcr.io on release
- **release.yml**: Automated releases via release-please

CodeQL SAST (python + actions) runs via GitHub's **default code-scanning setup**, not from a workflow file. Don't add a `codeql-action/init`/`analyze` job to ci.yml: GitHub rejects advanced-setup CodeQL uploads while default setup is enabled. ci.yml only uploads Ruff/Hadolint/Trivy SARIF to the same code-scanning dashboard.
