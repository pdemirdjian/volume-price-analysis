# ADR-0001: Run `apt-get upgrade` in the digest-pinned runtime image

**Status:** Accepted (2026-09-06)

## Context

The runtime stage of the `Dockerfile` starts from a digest-pinned
`python:3.14-slim` base and then runs `apt-get update && apt-get upgrade -y`.
The digest pin makes the *base* reproducible, but the upgrade pulls whatever
Debian security packages exist at build time, so two builds of the same commit
can differ (PDE-29 flagged this as worth a decision record).

CI runs a Trivy image scan on every build and Renovate bumps the base digest
when a new image is published. Debian security fixes, however, ship to the apt
repositories days or weeks before the upstream Python image is rebuilt, and
Trivy fails the build on HIGH/CRITICAL OS-level CVEs in that window.

## Decision

Keep `apt-get upgrade -y` in the runtime stage.

The image is a long-running daemon on a home cluster, not a distributed
artifact, so patch latency matters more than bit-for-bit reproducibility. The
digest pin still fixes the Python toolchain and everything under `/usr/local`;
only Debian packages float, and only forward.

## Consequences

- Builds of the same commit are not byte-identical; do not rely on image
  digests to prove provenance across rebuilds (the SBOM/provenance work in
  PDE-28 must attest the built image, not the recipe).
- Trivy stays green between base-image rebuilds without manual digest bumps.
- Revisit if the image is ever published for third-party consumption, where a
  reproducible build would be worth the CVE-window cost.
