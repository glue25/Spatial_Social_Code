# Release roadmap and TODO

This list distinguishes repository preparation from experiments that still need
work. An unchecked item is not a completed result or a commitment to a date.

## Repository preparation

- [x] Recover the organized ICLR submission snapshot and verify all 530 manifest entries.
- [x] Keep five benchmark entry points and preserve materially different historical protocols.
- [x] Add the six authors, affiliations, BibTeX and `CITATION.cff`.
- [x] Add a data-free, model-free Spatial/Social example.
- [x] Document installation, test commands, benchmark entry points and prerequisites.
- [x] Add CI configuration for offline checks and tests.
- [x] Add MIT terms for original SSC code and keep existing upstream notices.
- [x] Confirm the CI run passes on GitHub after pushing (Python 3.10 and 3.12).

- [x] Merge duplicate snapshots, move old analysis scripts to Git history and consolidate docs.
- [x] Replace vendored EnactToM source with a pinned download and verified SSC patch.

## Before public release

- [ ] **Third-party provenance:** identify the exact upstream revision and redistribution
  terms for MA-EgoQA helper files and the CoMind `cap-x` SAM launcher. The submitted
  package has no separate license notice for those two components. Preserve or add
  the applicable upstream notice; do not infer permission from a dependency name.
- [ ] **Paper link:** replace the pending-preprint note with the announced arXiv URL
  and identifier; keep author order consistent in README and CITATION.
- [ ] **Result map:** publish a compact table-to-artifact index with model, protocol,
  sample counts, split, seed, scorer version and input hashes. Keep historical
  and later corrected scoring policies separate.
- [ ] **Release tag:** create the first named release after the checklist is complete,
  with its source ZIP and checksum manifest.

## Reproduction improvements

- [ ] Add one end-to-end walkthrough for each benchmark, starting from lawfully
  acquired inputs and ending with the saved predictions and scoring command.
- [ ] Replace implicit experiment-directory assumptions with explicit data,
  config, output and service CLI arguments. Preserve historical defaults in
  documented snapshots rather than silently changing the reported protocol.
- [ ] Package the later symmetric CoMind adapter and its MA mechanism dependency,
  and the MA-EgoQA component-ablation dependency, if releasing those protocols.
  Do not present the submitted historical compiler as their replacement.
- [ ] Publish the required frozen manifests/configs for the MMToM and CoMind
  launchers. Their source code alone does not include every original run input.
- [ ] Validate a pinned Linux/CUDA/Habitat environment with a small licensed task;
  retain logs and distinguish an environment smoke test from a full evaluation.
- [x] Add one source-checkout command for demo, tests, integrity, dependency setup and runners.
- [ ] Support benchmark execution from an installed distribution without a source checkout.

## Research follow-up

- [ ] Freeze new evaluation splits and matching rules before starting additional runs.
- [ ] Record input evidence, tools, model versions, calls, token budgets and failures
  for Direct/SSC comparisons, including tool-matched controls where applicable.
- [ ] Report uncertainty across tasks/seeds and retain regressions and invalid outputs.
- [ ] Complete any human annotation or semantic-review study before reporting it.

These research items are future work, not experiments performed by the release cleanup.
