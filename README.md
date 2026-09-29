# Clinical SLM response evaluation

Code supporting **When Correct Labels Are Not Enough: Evaluating Small Language
Models for Clinical Text Analytics** (manuscript v0.13).

This release extends the original public code with the v0.13 follow-up experiments.
It contains only the implementation needed for the paper's experiments, analyses,
and results figure. It is not a copy of the full development project.

The study distinguishes correct labels from usable complete records, and new
model responses from answers recovered by changing response handling.

## What is included

- Initial synthetic criterion assessment, with and without JSON-format enforcement.
- A matched full-versus-compact response experiment: eight models, four conditions,
  387 requests per condition (12,384 new responses).
- Paired clinical-note annotation checking: eight models, two conditions,
  2,288 candidates per condition (36,608 new responses).
- Conservative Markdown-marker removal, state-only scoring, abstention/failure
  decomposition, paired clustered intervals, and model contrasts.
- Supporting CPGPrompt decision-tree traversal, strict/permissive handling, and
  punctuation-only replay.
- A label-only clinical-note baseline and the current all-model Figure 2 renderer.
- A saved-response comparison of normalized unenforced and enforced clinical-note
  label triples, separate from agreement with reference labels.

Dataset text, model outputs, patient-level artifacts, aggregates from the author's
runs, model weights, and manuscript files are **not distributed**. The author-made
conceptual Figure 1 is not an analysis output and is outside this code release.

## Start here

- [Reproduction guide](docs/REPRODUCING_V13.md): inputs, execution order, commands,
  and links from paper results to code.
- [Verification record](RELEASE_REVIEW.md): changes relative to the original release,
  checks performed, and their limitations.
- [Release boundary](RELEASE_SCOPE.md): inclusion, exclusions, and provenance.

```text
configs/        Portable task, model, prompt, wrapper, tree, and registry specifications
docs/           Reproduction instructions and retained protocol provenance
examples/       Invented, non-patient unit-test input
scripts/        Result reports, diagnostics, figure rendering, and release utilities
src/cpg2/       Execution, validation, scoring, and analysis dependency closure
tests/          Offline tests using invented fixtures and mocked generation
```

Historical identifiers such as E1, E1-S2, E2, and E3 remain in filenames to preserve
provenance; the reproduction guide maps them to the manuscript's task names.
Retained design documents describe earlier protocol decisions, not additional
experiments required for the current paper.

## Environments

Use Python 3.12. The tests require no model weights or patient data. NumPy is needed
for the new paired-analysis tests; Matplotlib is needed for figure rendering.
The current Figure 2 renderer also requires the system font Liberation Serif.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-analysis.txt
PYTHONPATH=src python -m unittest discover -s tests -t . -v
```

The current analysis pins are NumPy 2.4.6 and Matplotlib 3.10.9. Inference additionally
uses vLLM 0.22.1, Transformers 5.10.2, PyTorch 2.11.0, XGrammar 0.2.1, Accelerate
1.13.0, and huggingface-hub 1.18.0, as recorded for the matched follow-ups:

```bash
python -m pip install -r requirements-inference.txt
```

A compatible CUDA/GPU environment and the exact locally supplied model revisions
are required for inference. Wheel installation is platform-dependent. The paired
note runner verifies its pinned runtime and fails on a mismatch.

The older interaction-figure environment is retained separately in
`requirements-analysis-legacy.txt`. Use a separate environment if reproducing that
historical renderer with its original plotting dependencies; do not install the
legacy NumPy pin into the inference environment.

## Local inputs and integrity

Obtain CPGPrompt sources independently and access the MIMIC datasets under their
PhysioNet requirements. Place them at the ignored `data/` paths in the reproduction
guide. Place or symlink exact model snapshots at `models/<manifest_id>` as declared
in the model manifests. Runners load models locally; they do not download weights.

Portable configurations preserve prompts, model revisions, scientific settings,
and source-content hashes. Local paths and their configuration hashes necessarily
differ from the author's working copy. Manifest and source-audit hashes must be
bound to newly prepared local artifacts **before** a new run. Do not weaken hash
checks, edit a frozen run midway, or mix original and regenerated outputs.

A fresh rerun and a byte-for-byte replay of the author's archived outputs are
different workflows. This release supports the former; strict archival replay
requires the matching original code/configuration provenance and retained local
artifacts. Neither the test suite nor successful plotting alone establishes
numerical reproduction of the published results.

## Data attribution and access

- [CPGPrompt](https://github.com/bionlplab/CPGPrompt) and its
  [original publication](https://doi.org/10.1093/jamia/ocag026).
- [MIMIC-III-Ext-Notes v1.0.0](https://physionet.org/content/mimic-iii-ext-notes/1.0.0/),
  its [dataset DOI](https://doi.org/10.13026/9tfx-yx07), and the
  [original Liu et al. publication](https://arxiv.org/abs/2401.13588).
- [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/) and
  [database-description paper](https://doi.org/10.1038/sdata.2016.35).
- [PhysioNet platform attribution](https://doi.org/10.1038/s44360-026-00096-z).

Dataset access and redistribution remain governed by the upstream terms. Access
to this code does not grant access to clinical data or permission to redistribute
it. Keep all source data, instantiated prompts, responses, checkpoints, and logs
local. Aggregate exports also require privacy review; none are bundled here.

## Interpretation and reuse

These are exploratory component experiments on previously inspected datasets.
The clinical-note task is candidate annotation checking, not end-to-end concept
extraction or an identical-task transfer validation. Results do not establish
clinical safety, patient benefit, or measured performance at scale.

## License

Original contributions are copyright (c) 2026 UXFactory and licensed under the
[MIT License](LICENSE). CPGPrompt-derived trees, criterion registries, and related
target configurations retain BioNLP Lab's MIT copyright and license notice. See
[third-party notices](THIRD_PARTY_NOTICES.md) and the
[complete upstream license](LICENSES/CPGPrompt-MIT.txt).

This software license does not grant access to or redistribution rights for
separately obtained datasets, model weights, tokenizer assets, or dependencies.
Their respective terms continue to apply.
