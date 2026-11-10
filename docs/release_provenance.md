# Release provenance

The base is the `code/` directory in the downloaded ICLR supplementary archive
`17999_Thinking_with_Spatial_So_Supplementary Material.zip`.
All 530 entries in its original manifest were verified before preparation.

- [Source archive identity](provenance/source.json)
- [Original submitted file hashes](provenance/iclr_source_sha256.json)
- [Current release hashes](../SHA256SUMS.json)

The previous GitHub tree was a research archive. This preparation restores the
organized submission layout at the repository root. Previous files and commits
remain in Git history; no history rewrite is required.

Changes in this preparation are limited to public-facing documentation,
metadata, dependency declarations, a minimal example, CI and integrity tooling.
Benchmark implementations and their recorded scientific logic are preserved.
The exact differences from the submitted snapshot are listed in
`docs/provenance/source.json` under `modified_submission_files`.

No new benchmark predictions, simulator runs or paper scores were generated.
The new release manifest is separate from the immutable original manifest.
