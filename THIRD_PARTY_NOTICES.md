# Third-party notices and license scope

The root [MIT license](LICENSE) applies to original SSC code and documentation.
It does not override upstream licenses, grant dataset/model redistribution
rights or relicense vendored snapshots.

| Component | Location / retained notice |
| --- | --- |
| AutoToM interfaces | `third_party/autotom/`; [retained Apache-2.0 text](third_party/notices/upstream/AutoToM-LICENSE) |
| MMToM-QA integration | Benchmark-specific snapshots; [retained upstream license text](third_party/notices/upstream/MMToM-QA-Apache-2.0.txt) |
| EnactToM / Habitat dependencies | `benchmarks/habitat/external/EnactTom/`; [retained source notice](third_party/notices/EnactTom/LICENSE) and [EnactToM notice](third_party/notices/upstream/EnactToM-MIT.txt) |
| MA-EgoQA helpers | `benchmarks/ma_egoqa/egomas/` and historical copies; source project: https://github.com/KangsanKim07/MA-EgoQA |
| CoMind SAM service integration | `benchmarks/comind/cap-x/`; upstream attribution/terms to be completed before public release |

These notices are copied from the submitted package and retain their original
wording. In particular, a filename in that package is not independent proof of
license provenance for every copied file.

The submitted snapshot contains no separate MA-EgoQA or `cap-x` license notice.
Their exact upstream revisions and applicable redistribution terms remain an
explicit [release TODO](TODO.md). Those third-party portions are excluded from
the root MIT grant; do not assume they are MIT-licensed.

Original benchmark annotations, videos, simulator assets and model weights must
be obtained through their upstream projects under the applicable terms.
