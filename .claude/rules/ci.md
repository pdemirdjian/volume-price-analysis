# CI Pipeline

Key workflows in `.github/workflows/` (non-exhaustive):

- **ci.yml**: Test matrix (ubuntu/macos/windows), ruff lint+format, mypy, Trivy filesystem scan, Hadolint, CodeQL SAST, dependency-review (PRs only), Docker build + Trivy image scan
- **pr-title.yml**: Conventional commit PR title validation (`feat:`, `fix:`, `chore:`, etc.)
- **docker.yml**: Publish to ghcr.io on release
- **release.yml**: Automated releases via release-please
