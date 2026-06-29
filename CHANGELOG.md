# Changelog

## [2.6.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.32...v2.6.0) (2026-06-29)


### Features

* briefing signal-to-noise projection + additive MCP response headline (HOM-50) ([#326](https://github.com/pdemirdjian/volume-price-analysis/issues/326)) ([a40a219](https://github.com/pdemirdjian/volume-price-analysis/commit/a40a219fac95cee759072568b7a1e13149489e39))
* ground briefing output and flag hallucinated tickers (HOM-45) ([#323](https://github.com/pdemirdjian/volume-price-analysis/issues/323)) ([4424beb](https://github.com/pdemirdjian/volume-price-analysis/commit/4424bebce814e580eac3761449509379f1361a3e))
* label volatility as HV proxy and add scan skipped/errors diagnostics (HOM-39) ([#320](https://github.com/pdemirdjian/volume-price-analysis/issues/320)) ([03e491f](https://github.com/pdemirdjian/volume-price-analysis/commit/03e491f7b0ad105756a33225541fb938e826c82e))
* strictly-causal backtest / evidence harness (HOM-40) ([#321](https://github.com/pdemirdjian/volume-price-analysis/issues/321)) ([3399518](https://github.com/pdemirdjian/volume-price-analysis/commit/339951827aeaacc63bee831bc70de58582afed1d))


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.111.0 ([#329](https://github.com/pdemirdjian/volume-price-analysis/issues/329)) ([46db124](https://github.com/pdemirdjian/volume-price-analysis/commit/46db1243dba36638843c51c5fd9b167fa5d2a38a))
* **deps:** update dependency anthropic to ~=0.112.0 ([#335](https://github.com/pdemirdjian/volume-price-analysis/issues/335)) ([08d49dd](https://github.com/pdemirdjian/volume-price-analysis/commit/08d49dd5a7a99d3f5e65450906c370fd310c8d8e))
* **deps:** update dependency google-genai to ~=2.10.0 ([#334](https://github.com/pdemirdjian/volume-price-analysis/issues/334)) ([b99cbca](https://github.com/pdemirdjian/volume-price-analysis/commit/b99cbca884878458a6be92d8d99f9fd371378619))
* **deps:** update dependency google-genai to ~=2.9.0 ([#330](https://github.com/pdemirdjian/volume-price-analysis/issues/330)) ([b95c7bb](https://github.com/pdemirdjian/volume-price-analysis/commit/b95c7bb41a93defc6246ba1bd018e793c45da185))
* **deps:** update dependency numpy to v2.5.0 ([#333](https://github.com/pdemirdjian/volume-price-analysis/issues/333)) ([9df1cd3](https://github.com/pdemirdjian/volume-price-analysis/commit/9df1cd3e53b3bdd1c733314c349266cb7f78a50e))
* guard enhanced_volume_profile against empty-frame IndexError ([#319](https://github.com/pdemirdjian/volume-price-analysis/issues/319)) ([e237fca](https://github.com/pdemirdjian/volume-price-analysis/commit/e237fcad439dbef0d9a59b16411dde9887d33a32))
* harden indicator edge cases (HOM-37) ([#318](https://github.com/pdemirdjian/volume-price-analysis/issues/318)) ([45c0564](https://github.com/pdemirdjian/volume-price-analysis/commit/45c05642b8538c9f0f28a1373243dc4e085fbe04))
* make scan ADX coherent with the composite score's adaptive ADX ([#325](https://github.com/pdemirdjian/volume-price-analysis/issues/325)) ([a4c7e55](https://github.com/pdemirdjian/volume-price-analysis/commit/a4c7e554be2b59512011bdffc516b56d3fa7cc49))

## [2.5.32](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.31...v2.5.32) (2026-06-20)


### Bug Fixes

* **deps:** update dependency mcp to v1.28.0 ([#313](https://github.com/pdemirdjian/volume-price-analysis/issues/313)) ([c85cbca](https://github.com/pdemirdjian/volume-price-analysis/commit/c85cbca550aba80131304e72058b587ccdbf0905))

## [2.5.31](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.30...v2.5.31) (2026-06-12)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.109.1 ([#306](https://github.com/pdemirdjian/volume-price-analysis/issues/306)) ([4feeca7](https://github.com/pdemirdjian/volume-price-analysis/commit/4feeca795d3d69e16b38202f1771a75c279074f2))
* **deps:** update ghcr.io/astral-sh/uv docker tag to v0.11.21 ([#304](https://github.com/pdemirdjian/volume-price-analysis/issues/304)) ([f30cf51](https://github.com/pdemirdjian/volume-price-analysis/commit/f30cf51eb0e3bd2f9fb3f7f52f2629b83da0cbe2))
* **deps:** update python docker tag to v3.14.6 ([#305](https://github.com/pdemirdjian/volume-price-analysis/issues/305)) ([921fad8](https://github.com/pdemirdjian/volume-price-analysis/commit/921fad8d6371ff04991ee44d17b7007509a07c80))

## [2.5.30](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.29...v2.5.30) (2026-06-09)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.106.0 ([#301](https://github.com/pdemirdjian/volume-price-analysis/issues/301)) ([5792948](https://github.com/pdemirdjian/volume-price-analysis/commit/579294846d022ab48706f6137f3930d52f8ed3fc))
* **deps:** update dependency anthropic to ~=0.107.0 ([#302](https://github.com/pdemirdjian/volume-price-analysis/issues/302)) ([f92f3e3](https://github.com/pdemirdjian/volume-price-analysis/commit/f92f3e37e165b8ec76ad85d1657843b83e5d3745))
* **deps:** update dependency google-genai to ~=2.8.0 ([#297](https://github.com/pdemirdjian/volume-price-analysis/issues/297)) ([dac8651](https://github.com/pdemirdjian/volume-price-analysis/commit/dac86517eecb8e2d7327229af202024539ff54b4))

## [2.5.29](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.28...v2.5.29) (2026-06-01)


### Bug Fixes

* **deps:** update dependency mcp to v1.27.2 ([#291](https://github.com/pdemirdjian/volume-price-analysis/issues/291)) ([47fccaf](https://github.com/pdemirdjian/volume-price-analysis/commit/47fccaf4f599e5b276af99e930e6e2d3849a5b63))

## [2.5.28](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.27...v2.5.28) (2026-05-31)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.105.0 ([#287](https://github.com/pdemirdjian/volume-price-analysis/issues/287)) ([3c468e4](https://github.com/pdemirdjian/volume-price-analysis/commit/3c468e4c0d345f8706e146fe66a4627fb03b9015))
* **deps:** update dependency google-genai to ~=2.7.0 ([#288](https://github.com/pdemirdjian/volume-price-analysis/issues/288)) ([13ed28e](https://github.com/pdemirdjian/volume-price-analysis/commit/13ed28ee686767bf57647f7fc797d776fd6800ee))
* **deps:** update dependency yfinance to v1.4.1 ([#290](https://github.com/pdemirdjian/volume-price-analysis/issues/290)) ([593178b](https://github.com/pdemirdjian/volume-price-analysis/commit/593178b0316923aed436ff804e7eb02f95716478))

## [2.5.27](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.26...v2.5.27) (2026-05-26)


### Bug Fixes

* **deps:** update dependency google-genai to ~=2.6.0 ([#280](https://github.com/pdemirdjian/volume-price-analysis/issues/280)) ([20aa28f](https://github.com/pdemirdjian/volume-price-analysis/commit/20aa28f12b69e2fb938d2acaeb4daa499807d9a8))
* **deps:** update dependency yfinance to v1.4.0 ([#284](https://github.com/pdemirdjian/volume-price-analysis/issues/284)) ([e1bea16](https://github.com/pdemirdjian/volume-price-analysis/commit/e1bea16f30c5e92c0fc8524a1239aef3f2062414))

## [2.5.26](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.25...v2.5.26) (2026-05-24)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.104.0 ([#279](https://github.com/pdemirdjian/volume-price-analysis/issues/279)) ([4e54a88](https://github.com/pdemirdjian/volume-price-analysis/commit/4e54a889c32bbe96243a70fd2dc25257efd43ddc))
* **deps:** update dependency google-genai to ~=2.5.0 ([#277](https://github.com/pdemirdjian/volume-price-analysis/issues/277)) ([9de86bc](https://github.com/pdemirdjian/volume-price-analysis/commit/9de86bca5004d8c19a0ee782c75654cb4205b45b))

## [2.5.25](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.24...v2.5.25) (2026-05-23)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.103.0 ([#270](https://github.com/pdemirdjian/volume-price-analysis/issues/270)) ([0e8337c](https://github.com/pdemirdjian/volume-price-analysis/commit/0e8337cc8574b3f3dbc1de5b92539f7b6a2eaf5b))

## [2.5.24](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.23...v2.5.24) (2026-05-19)


### Bug Fixes

* **deps:** update dependency numpy to v2.4.6 ([#267](https://github.com/pdemirdjian/volume-price-analysis/issues/267)) ([c422dbc](https://github.com/pdemirdjian/volume-price-analysis/commit/c422dbcdb33d1fd1eca108f7ef3e23350d0b5c5a))

## [2.5.23](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.22...v2.5.23) (2026-05-19)


### Bug Fixes

* **deps:** update dependency google-genai to ~=2.3.0 ([#260](https://github.com/pdemirdjian/volume-price-analysis/issues/260)) ([956cd0e](https://github.com/pdemirdjian/volume-price-analysis/commit/956cd0e128e616f6c2b2b0dc9b6a41b96a6c5709))
* **deps:** update dependency google-genai to ~=2.4.0 ([#266](https://github.com/pdemirdjian/volume-price-analysis/issues/266)) ([1b512ba](https://github.com/pdemirdjian/volume-price-analysis/commit/1b512ba93b2df1e282edeaafa40fe95ab98d137f))
* **deps:** update dependency numpy to v2.4.5 ([#263](https://github.com/pdemirdjian/volume-price-analysis/issues/263)) ([000724f](https://github.com/pdemirdjian/volume-price-analysis/commit/000724f975a3750eec21416f48ed22b236d6e3fb))
* **deps:** update ghcr.io/astral-sh/uv docker tag to v0.11.15 ([#265](https://github.com/pdemirdjian/volume-price-analysis/issues/265)) ([37f3536](https://github.com/pdemirdjian/volume-price-analysis/commit/37f3536cd2ba5ffb0f069d538f352e5cc2e328a3))

## [2.5.22](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.21...v2.5.22) (2026-05-17)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.102.0 ([#257](https://github.com/pdemirdjian/volume-price-analysis/issues/257)) ([1e9efda](https://github.com/pdemirdjian/volume-price-analysis/commit/1e9efda250defdf2dfd9d8eb048694624dc0e2df))
* **deps:** update python docker tag to v3.14.5 ([#259](https://github.com/pdemirdjian/volume-price-analysis/issues/259)) ([bc8eeac](https://github.com/pdemirdjian/volume-price-analysis/commit/bc8eeacd8bdd88642ef77e99370170dbbfcb176e))

## [2.5.21](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.20...v2.5.21) (2026-05-16)


### Bug Fixes

* **deps:** update dependency google-genai to ~=2.1.0 ([#253](https://github.com/pdemirdjian/volume-price-analysis/issues/253)) ([7016f50](https://github.com/pdemirdjian/volume-price-analysis/commit/7016f5071b3efcc9a8c3859b230f776ae7b1c9a7))
* **deps:** update dependency google-genai to ~=2.2.0 ([#256](https://github.com/pdemirdjian/volume-price-analysis/issues/256)) ([32db8d0](https://github.com/pdemirdjian/volume-price-analysis/commit/32db8d00adbe4a43101877bb0980e8e519b66137))

## [2.5.20](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.19...v2.5.20) (2026-05-15)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.101.0 ([#250](https://github.com/pdemirdjian/volume-price-analysis/issues/250)) ([bac8aad](https://github.com/pdemirdjian/volume-price-analysis/commit/bac8aad48289139928f9bd930d0b5af65d8d5810))
* **deps:** update dependency pandas to v3.0.3 ([#252](https://github.com/pdemirdjian/volume-price-analysis/issues/252)) ([5109481](https://github.com/pdemirdjian/volume-price-analysis/commit/51094817b22a71747ddab8c74fd1ac92866b36ab))

## [2.5.19](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.18...v2.5.19) (2026-05-11)


### Bug Fixes

* **deps:** update ghcr.io/astral-sh/uv docker tag to v0.11.13 ([#246](https://github.com/pdemirdjian/volume-price-analysis/issues/246)) ([5230e9e](https://github.com/pdemirdjian/volume-price-analysis/commit/5230e9ec1c3a0c84feaaf9b608cc538ddc7f3964))

## [2.5.18](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.17...v2.5.18) (2026-05-09)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.100.0 ([#237](https://github.com/pdemirdjian/volume-price-analysis/issues/237)) ([ba31000](https://github.com/pdemirdjian/volume-price-analysis/commit/ba310001a45fdaa3c5768ac9775e865745be240e))
* **deps:** update dependency google-genai to v2 ([#240](https://github.com/pdemirdjian/volume-price-analysis/issues/240)) ([7fc2389](https://github.com/pdemirdjian/volume-price-analysis/commit/7fc23890b95c52704ff68633ef9eb3b6496e3eea))
* **deps:** update dependency mcp to v1.27.1 ([#235](https://github.com/pdemirdjian/volume-price-analysis/issues/235)) ([c9f54db](https://github.com/pdemirdjian/volume-price-analysis/commit/c9f54db994b18033f89583ddb628a22a9f8bd331))
* **deps:** update ghcr.io/astral-sh/uv docker tag to v0.11.12 ([#236](https://github.com/pdemirdjian/volume-price-analysis/issues/236)) ([a84e5a8](https://github.com/pdemirdjian/volume-price-analysis/commit/a84e5a8baf941d800a376af3d09b8cfb13a65464))

## [2.5.17](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.16...v2.5.17) (2026-05-08)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.98.1 ([#230](https://github.com/pdemirdjian/volume-price-analysis/issues/230)) ([4e9e3f4](https://github.com/pdemirdjian/volume-price-analysis/commit/4e9e3f4dbc133a98c4e7bccfffe2c48f4f5d6878))
* **deps:** update dependency anthropic to ~=0.99.0 ([#233](https://github.com/pdemirdjian/volume-price-analysis/issues/233)) ([40c034f](https://github.com/pdemirdjian/volume-price-analysis/commit/40c034fd0a5dcb2b9d4beb0298cc57f2b605f126))
* **deps:** update dependency google-genai to ~=1.75.0 ([#232](https://github.com/pdemirdjian/volume-price-analysis/issues/232)) ([97a4e38](https://github.com/pdemirdjian/volume-price-analysis/commit/97a4e38560fbac3f12c9e75e12b2639d1fe9f6a1))

## [2.5.16](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.15...v2.5.16) (2026-05-04)


### Bug Fixes

* **deps:** update dependency google-genai to ~=1.74.0 ([#226](https://github.com/pdemirdjian/volume-price-analysis/issues/226)) ([b2b7688](https://github.com/pdemirdjian/volume-price-analysis/commit/b2b76885df3f1733f802aa5e92f920a145c4ed24))

## [2.5.15](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.14...v2.5.15) (2026-04-29)


### Bug Fixes

* **deps:** update ghcr.io/astral-sh/uv docker tag to v0.11.8 ([#224](https://github.com/pdemirdjian/volume-price-analysis/issues/224)) ([07fc7af](https://github.com/pdemirdjian/volume-price-analysis/commit/07fc7afa73d3a6d00f00e7fba1c054e14cf4b5e7))

## [2.5.14](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.13...v2.5.14) (2026-04-24)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.97.0 ([#220](https://github.com/pdemirdjian/volume-price-analysis/issues/220)) ([fbb64d7](https://github.com/pdemirdjian/volume-price-analysis/commit/fbb64d772f3e541dbc35b406a0764b8d59f626e2))
* **deps:** update python docker tag to v3.14.4 ([#218](https://github.com/pdemirdjian/volume-price-analysis/issues/218)) ([8fc2a78](https://github.com/pdemirdjian/volume-price-analysis/commit/8fc2a7894d42c9ef7a99925a7755ddb1be41c1ad))

## [2.5.13](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.12...v2.5.13) (2026-04-20)


### Bug Fixes

* **deps:** update non-major dependencies ([#207](https://github.com/pdemirdjian/volume-price-analysis/issues/207)) ([57a0b44](https://github.com/pdemirdjian/volume-price-analysis/commit/57a0b4417d290709501e500544ed50aae31f605b))

## [2.5.12](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.11...v2.5.12) (2026-04-13)


### Bug Fixes

* **renovate:** revert lock file maintenance to chore commit type ([#203](https://github.com/pdemirdjian/volume-price-analysis/issues/203)) ([d6fd280](https://github.com/pdemirdjian/volume-price-analysis/commit/d6fd28048848705c0987d6784660556e39780ad4))

## [2.5.11](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.10...v2.5.11) (2026-04-13)


### Bug Fixes

* **deps:** lock file maintenance ([#196](https://github.com/pdemirdjian/volume-price-analysis/issues/196)) ([6baf4a6](https://github.com/pdemirdjian/volume-price-analysis/commit/6baf4a60ede97a9838a33167b9774f9346587b71))
* **deps:** update python:3.14-slim docker digest to bc389f7 ([#194](https://github.com/pdemirdjian/volume-price-analysis/issues/194)) ([3fa0558](https://github.com/pdemirdjian/volume-price-analysis/commit/3fa0558563f304d3e297c91adfd0c870e47490bd))
* **renovate:** use fix commit type for container-affecting updates ([#198](https://github.com/pdemirdjian/volume-price-analysis/issues/198)) ([70a5aff](https://github.com/pdemirdjian/volume-price-analysis/commit/70a5aff297f9292c88734d93705f262598b7050c))

## [2.5.10](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.9...v2.5.10) (2026-04-11)


### Bug Fixes

* **deps:** update non-major dependencies ([#191](https://github.com/pdemirdjian/volume-price-analysis/issues/191)) ([88ac619](https://github.com/pdemirdjian/volume-price-analysis/commit/88ac619f5314fdc50fa81bafda3e528839a266b8))

## [2.5.9](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.8...v2.5.9) (2026-04-07)


### Bug Fixes

* **deps:** upgrade curl-cffi to 0.15.0 to patch CVE-2026-33752 ([#184](https://github.com/pdemirdjian/volume-price-analysis/issues/184)) ([d9b2b96](https://github.com/pdemirdjian/volume-price-analysis/commit/d9b2b96839b8f0cbd5ce2732f02956e0a6136425))

## [2.5.8](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.7...v2.5.8) (2026-04-06)


### Bug Fixes

* **deps:** update non-major dependencies ([#181](https://github.com/pdemirdjian/volume-price-analysis/issues/181)) ([4c2ef80](https://github.com/pdemirdjian/volume-price-analysis/commit/4c2ef804a10da1aa8682bfafa435ec30817e7315))

## [2.5.7](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.6...v2.5.7) (2026-04-02)


### Bug Fixes

* **deps:** update non-major dependencies ([#176](https://github.com/pdemirdjian/volume-price-analysis/issues/176)) ([29ebb08](https://github.com/pdemirdjian/volume-price-analysis/commit/29ebb0809f42b04c39798e324c014ffe59835976))

## [2.5.6](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.5...v2.5.6) (2026-04-01)


### Bug Fixes

* **deps:** update dependency anthropic to ~=0.87.0 [security] ([#172](https://github.com/pdemirdjian/volume-price-analysis/issues/172)) ([a16b2f4](https://github.com/pdemirdjian/volume-price-analysis/commit/a16b2f482e150f74fbdd54702f5888d24db21029))

## [2.5.5](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.4...v2.5.5) (2026-03-30)


### Bug Fixes

* **deps:** update non-major dependencies ([#170](https://github.com/pdemirdjian/volume-price-analysis/issues/170)) ([f98fd02](https://github.com/pdemirdjian/volume-price-analysis/commit/f98fd02de73a67e8e5c96626a87e9015dac7c204))

## [2.5.4](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.3...v2.5.4) (2026-03-26)


### Bug Fixes

* replace abandoned actions/delete-package-versions with gh CLI ([#167](https://github.com/pdemirdjian/volume-price-analysis/issues/167)) ([a5d7b15](https://github.com/pdemirdjian/volume-price-analysis/commit/a5d7b155f5cd3f5fb1dccfb3c67fdd4d17bb20e4))

## [2.5.3](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.2...v2.5.3) (2026-03-20)


### Bug Fixes

* **deps:** update non-major dependencies ([#150](https://github.com/pdemirdjian/volume-price-analysis/issues/150)) ([9c8f8ce](https://github.com/pdemirdjian/volume-price-analysis/commit/9c8f8ceecf85d32141a29898a9f0d07235b23d16))
* **deps:** update non-major dependencies ([#157](https://github.com/pdemirdjian/volume-price-analysis/issues/157)) ([7786372](https://github.com/pdemirdjian/volume-price-analysis/commit/778637265eec42f5b2c1558d7ead3c5ff7f6ff1c))
* **docker:** patch OS-level CVEs in base image ([#160](https://github.com/pdemirdjian/volume-price-analysis/issues/160)) ([3bdc109](https://github.com/pdemirdjian/volume-price-analysis/commit/3bdc10980fa1f5976243b2f7d335263dace34b80))

## [2.5.2](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.1...v2.5.2) (2026-03-08)


### Bug Fixes

* use github app token for release-please ([#144](https://github.com/pdemirdjian/volume-price-analysis/issues/144)) ([e8b0ab4](https://github.com/pdemirdjian/volume-price-analysis/commit/e8b0ab4d137bb259fa1a2a73600ed1ee0b1bfd4c))

## [2.5.1](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.5.0...v2.5.1) (2026-03-07)


### Bug Fixes

* **deps:** update non-major dependencies ([#138](https://github.com/pdemirdjian/volume-price-analysis/issues/138)) ([129842d](https://github.com/pdemirdjian/volume-price-analysis/commit/129842dd8bdb77b6a32684d679bc5e914d482f3b))

## [2.5.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.4.3...v2.5.0) (2026-03-07)


### Features

* add team review agents and fix 19 issues across all modules ([#130](https://github.com/pdemirdjian/volume-price-analysis/issues/130)) ([93ae820](https://github.com/pdemirdjian/volume-price-analysis/commit/93ae820613715a5483b5744e3170e240e5cbbb1b))

## [2.4.3](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.4.2...v2.4.3) (2026-03-03)


### Bug Fixes

* **deps:** update dependency nh3 to ~=0.3.3 ([#127](https://github.com/pdemirdjian/volume-price-analysis/issues/127)) ([92d681c](https://github.com/pdemirdjian/volume-price-analysis/commit/92d681c58b68daf029a05397ef4f35eca322ee79))

## [2.4.2](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.4.1...v2.4.2) (2026-03-03)


### Bug Fixes

* comprehensive security hardening from audit ([#125](https://github.com/pdemirdjian/volume-price-analysis/issues/125)) ([88c3846](https://github.com/pdemirdjian/volume-price-analysis/commit/88c38468e5783e3611a16a12dca21571f3b22d7d))

## [2.4.1](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.4.0...v2.4.1) (2026-03-03)


### Bug Fixes

* split comma-separated EMAIL_TO for multi-recipient delivery ([#122](https://github.com/pdemirdjian/volume-price-analysis/issues/122)) ([576b36e](https://github.com/pdemirdjian/volume-price-analysis/commit/576b36ed4f05e4db9b2f7e60c272f9cbebc9652c))

## [2.4.0](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.3.4...v2.4.0) (2026-03-02)


### Features

* add standalone MCP tools for A/D Line and CMF indicators ([#119](https://github.com/pdemirdjian/volume-price-analysis/issues/119)) ([59d9d10](https://github.com/pdemirdjian/volume-price-analysis/commit/59d9d10ddcba948c0f35a2e3d47a4e46a3dde1e6))

## [2.3.4](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.3.3...v2.3.4) (2026-03-01)


### Bug Fixes

* increase AI max_tokens to prevent briefing truncation and add 14-day holding period context ([8812cfd](https://github.com/pdemirdjian/volume-price-analysis/commit/8812cfd25e79c7c09e10bf4a8da76cf4fb50568c))
* prevent briefing truncation and add 14-day holding period context ([#114](https://github.com/pdemirdjian/volume-price-analysis/issues/114)) ([4738547](https://github.com/pdemirdjian/volume-price-analysis/commit/47385477eb178b3aae04b4e1e44a3ad1e43af796))

## [2.3.3](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.3.2...v2.3.3) (2026-02-28)


### Bug Fixes

* remove vulnerable system pip from Docker runtime image ([#111](https://github.com/pdemirdjian/volume-price-analysis/issues/111)) ([8518569](https://github.com/pdemirdjian/volume-price-analysis/commit/85185690ede05b1b4bb2aa5e0dca0a72dfa56e58))

## [2.3.2](https://github.com/pdemirdjian/volume-price-analysis/compare/v2.3.1...v2.3.2) (2026-02-28)


### Bug Fixes

* address Copilot review feedback on Wilder's smoothing and docs ([#110](https://github.com/pdemirdjian/volume-price-analysis/issues/110)) ([b69d893](https://github.com/pdemirdjian/volume-price-analysis/commit/b69d8939bcd2a4cb7c3ef03df49047853912a8e8))
* correct Wilder's smoothing and crash bugs in indicators ([#108](https://github.com/pdemirdjian/volume-price-analysis/issues/108)) ([d1b43ff](https://github.com/pdemirdjian/volume-price-analysis/commit/d1b43ffd382cdb42df7978922b3e8915227df072))

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
