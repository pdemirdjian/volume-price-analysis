# Changelog

## [2.3.1](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.3.0...v2.3.1) (2026-02-24)


### Bug Fixes

* **deps:** update non-major dependencies ([#102](https://github.com/pdemirdjian/volume-price-analysis/issues/102)) ([7bb9e46](https://github.com/pdemirdjian/volume-price-analysis/commit/7bb9e4670b165dc6a7dcc1c28bf9a4e4644d375e))

## [2.3.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.2.1...v2.3.0) (2026-02-17)


### Features

* add Trivy, Hadolint, and pre-commit hooks ([#95](https://github.com/pdemirdjian/volume-price-analysis/issues/95)) ([ea6fed5](https://github.com/pdemirdjian/volume-price-analysis/commit/ea6fed5323599e4cf855f9a63d2c9909f314f2a9))

## [2.2.1](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.2.0...v2.2.1) (2026-02-17)


### Bug Fixes

* correct Renovate dev deps automerge rule ([#93](https://github.com/pdemirdjian/volume-price-analysis/issues/93)) ([f794df7](https://github.com/pdemirdjian/volume-price-analysis/commit/f794df77c999567579919c7faad00dc3384550de))

## [2.2.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.1.1...v2.2.0) (2026-02-17)


### Features

* dynamic S&P 500 symbol universes via pytickersymbols ([#90](https://github.com/pdemirdjian/volume-price-analysis/issues/90)) ([a5bd295](https://github.com/pdemirdjian/volume-price-analysis/commit/a5bd29556bad9f77d51184aa4ee0ce803425bdfc))

## [2.1.1](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.1.0...v2.1.1) (2026-02-17)


### Bug Fixes

* **deps:** update non-major dependencies ([#87](https://github.com/pdemirdjian/volume-price-analysis/issues/87)) ([af8b31b](https://github.com/pdemirdjian/volume-price-analysis/commit/af8b31ba6996626dcd042557874c9d1ddbe534f4))

## [2.1.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.0.0...v2.1.0) (2026-02-17)


### Features

* add cross-platform support for macOS, Windows, and Linux ([#86](https://github.com/pdemirdjian/volume-price-analysis/issues/86)) ([6d30764](https://github.com/pdemirdjian/volume-price-analysis/commit/6d307641916fb64e3095209583a4dfee070002c5))

## [2.0.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.4.3...v2.0.0) (2026-02-16)


### ⚠ BREAKING CHANGES

* replace supercronic with Python asyncio scheduler ([#80](https://github.com/pdemirdjian/volume-price-analysis/issues/80))

### Features

* replace supercronic with Python asyncio scheduler ([#80](https://github.com/pdemirdjian/volume-price-analysis/issues/80)) ([57291e1](https://github.com/pdemirdjian/volume-price-analysis/commit/57291e1886cdb249bc8c41551f7ac77efc2249e7))

## [1.4.3](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.4.2...v1.4.3) (2026-02-16)


### Bug Fixes

* validate integer parameters in MCP tool handlers ([#78](https://github.com/pdemirdjian/volume-price-analysis/issues/78)) ([2f252de](https://github.com/pdemirdjian/volume-price-analysis/commit/2f252de73ad207d336a2625ea653b66a898d20a1))

## [1.4.2](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.4.1...v1.4.2) (2026-02-15)


### Bug Fixes

* mirror CI status checks to release-please PRs ([#73](https://github.com/pdemirdjian/volume-price-analysis/issues/73)) ([c5bb9f3](https://github.com/pdemirdjian/volume-price-analysis/commit/c5bb9f3cb702a21ef590bae06aa6adb1f3a712db))

## [1.4.1](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.4.0...v1.4.1) (2026-02-15)


### Bug Fixes

* tag Docker images with version on releases ([#71](https://github.com/pdemirdjian/volume-price-analysis/issues/71)) ([45195e6](https://github.com/pdemirdjian/volume-price-analysis/commit/45195e6f3dbbdbe6dd0ff082544329da2870c1c8))

## [1.4.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.3.0...v1.4.0) (2026-02-15)


### Features

* support Gemini and Anthropic as AI providers ([#68](https://github.com/pdemirdjian/volume-price-analysis/issues/68)) ([15f669c](https://github.com/pdemirdjian/volume-price-analysis/commit/15f669c22dd08a5586e74f1094ea90444f7654ff))

## [1.3.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.2.0...v1.3.0) (2026-02-15)


### Features

* add multi-arch Docker builds (amd64 + arm64) ([#66](https://github.com/pdemirdjian/volume-price-analysis/issues/66)) ([df6dc00](https://github.com/pdemirdjian/volume-price-analysis/commit/df6dc0051060df3a236e3881ff60063a7cf6d4a7))

## [1.2.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.1.3...v1.2.0) (2026-02-15)


### Features

* add morning briefing agent with Docker deployment ([#57](https://github.com/pdemirdjian/volume-price-analysis/issues/57)) ([569d627](https://github.com/pdemirdjian/volume-price-analysis/commit/569d62701c0ba44493cae0ee09946736cb92cf5a))


### Bug Fixes

* fix Docker build and add CI build test ([#60](https://github.com/pdemirdjian/volume-price-analysis/issues/60)) ([eec027f](https://github.com/pdemirdjian/volume-price-analysis/commit/eec027fecb1047e451ff49b49cdcecb5d0e15e2f))
* only tag Docker image as latest on version releases ([#63](https://github.com/pdemirdjian/volume-price-analysis/issues/63)) ([f28a24f](https://github.com/pdemirdjian/volume-price-analysis/commit/f28a24f5ce8f2d770508d2267c900cf6aea46cfa))

## [1.1.3](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.1.2...v1.1.3) (2026-02-02)


### Bug Fixes

* **deps:** update non-major dependencies ([#45](https://github.com/pdemirdjian/volume-price-analysis/issues/45)) ([0f3c65b](https://github.com/pdemirdjian/volume-price-analysis/commit/0f3c65b75b23c37d5884bfc90e1dc7b88706fe89))

## [1.1.2](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.1.1...v1.1.2) (2026-01-30)


### Bug Fixes

* **deps:** update dependency pandas to v3 ([#41](https://github.com/pdemirdjian/volume-price-analysis/issues/41)) ([9da1af9](https://github.com/pdemirdjian/volume-price-analysis/commit/9da1af94aa836f62c1ec69c63662b4f4fdbc4481))
* **deps:** update non-major dependencies ([#39](https://github.com/pdemirdjian/volume-price-analysis/issues/39)) ([755fba2](https://github.com/pdemirdjian/volume-price-analysis/commit/755fba2c9fd42869de6d3ba11fd0e2e16c671d44))

## [1.1.1](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.1.0...v1.1.1) (2026-01-17)


### Documentation

* fix markdown linting errors across all documentation ([#29](https://github.com/pdemirdjian/volume-price-analysis/issues/29)) ([f2dc2b6](https://github.com/pdemirdjian/volume-price-analysis/commit/f2dc2b6a33f08e65382f44da4fe110c24c47dbe2))

## [1.1.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.0.2...v1.1.0) (2026-01-17)

### Features

* improve reliability and performance
  ([#26](https://github.com/pdemirdjian/volume-price-analysis/issues/26))
  ([4cca0d1](https://github.com/pdemirdjian/volume-price-analysis/commit/4cca0d134dec2e2ffde42c78561630d771de7713))

## [1.0.2](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.0.1...v1.0.2) (2026-01-16)

### Bug Fixes

* fix indicator count in CLAUDE.md
  ([#20](https://github.com/pdemirdjian/volume-price-analysis/issues/20))
  ([339cad7](https://github.com/pdemirdjian/volume-price-analysis/commit/339cad73bc65a62d30a9fd6ff548b1db572df476))

## [1.0.1](https://github.com/pdemirdjian/volume-price-analysis/compare/v1.0.0...v1.0.1) (2026-01-16)

### Docs

* add security policy
  ([#17](https://github.com/pdemirdjian/volume-price-analysis/issues/17))
  ([22830eb](https://github.com/pdemirdjian/volume-price-analysis/commit/22830eb8cb8918e0999007cca60db870bd386b45))
