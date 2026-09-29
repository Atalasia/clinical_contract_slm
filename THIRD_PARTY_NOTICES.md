# Third-party notices

Original contributions to this release are copyright (c) 2026 UXFactory and
licensed under the [MIT License](LICENSE). Upstream material retains its own
copyright notice, as identified below.

## CPGPrompt-derived configurations

- Source: [BioNLP Lab's CPGPrompt](https://github.com/bionlplab/CPGPrompt).
- Upstream license: [MIT](https://github.com/bionlplab/CPGPrompt/blob/main/LICENSE),
  checked 2026-09-29.
- Upstream copyright: Copyright (c) 2025 BioNLP Lab.
- Complete upstream license: [LICENSES/CPGPrompt-MIT.txt](LICENSES/CPGPrompt-MIT.txt).

The following files retain or adapt CPGPrompt tree structure, criterion wording,
or terminal actions. The source names below are the local source references
recorded in this project's configurations, not asserted upstream repository paths.

| Local source reference | Included derived files |
|---|---|
| `trees/headache_tree.json` | `configs/trees/headache_redflags.normalized.json`; `configs/criteria/headache_redflags.registry.json`; `configs/p2/headache.targets.v1.json` |
| `trees/lbp_tree.json` | `configs/trees/low_back_pain.normalized.json`; `configs/criteria/low_back_pain.registry.json`; `configs/p2/low_back_pain.targets.v1.json` |
| `trees/pc_tree.json` | `configs/trees/prostate_cancer.normalized.json`; `configs/criteria/prostate_cancer.registry.json`; `configs/p2/prostate_cancer.targets.v1.json` |

Project adaptations include normalized node/edge representations, stable
criterion identifiers, assessment questions, evidence-state definitions, and
related target/retrieval specifications. UXFactory's notice does not replace
BioNLP Lab's notice for the upstream portions. Preserve the upstream notice and
complete MIT license when redistributing these adaptations.

The historical binary prompt and traversal experiment also build on the
CPGPrompt approach; they are not a claim of authorship of that approach.
This notice does not grant rights to separately obtained guideline documents.

## Dataset attribution and access

The prompt in `configs/slm_benchmark/prompts/ext_notes_context.v1.json` implements
the label definitions documented by
[MIMIC-III-Ext-Notes v1.0.0](https://physionet.org/content/mimic-iii-ext-notes/1.0.0/).
It contains placeholders, not clinical notes or dataset records. Dataset and
publication citations are retained in [README.md](README.md#data-attribution-and-access).

MIMIC data, source synthetic datasets, instantiated patient prompts, and model
responses are not distributed. The MIT software license does not license these
separately obtained resources or change their access and redistribution terms.

## Separately installed models and dependencies

Model manifests identify separately obtained models and their declared licenses.
Model weights, tokenizer assets, and model-specific chat templates are not
bundled; the code loads them from user-supplied local snapshots. Third-party
Python libraries are installed dependencies, not vendored source in this release.
Those resources retain their respective licenses and terms; this release's MIT
license does not replace them.
