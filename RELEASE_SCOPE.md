# Release scope: manuscript v0.13

This release extends the original public commit
`2603548` (`Make release configuration self-contained`). It preserves the narrow
release boundary: implementation responsible for paper results, not the complete
research workspace.

## Included

- Portable benchmark/model manifests, frozen model revisions, prompts, wrappers,
  output schemas, and synthetic task construction specifications.
- Original synthetic assessment, initial JSON-enforcement comparison, and
  supporting binary CPGPrompt tree traversal.
- The four-condition matched synthetic experiment:
  `slm_compact_contract.py`, `slm_contract_ablation.py`, and
  `slm_contract_analysis.py`.
- The paired clinical-note experiment:
  `slm_e3_contract.py` and `slm_e3_contract_analysis.py`.
- Conservative saved-output normalization and failure decomposition:
  `slm_contract_sensitivity.py` and its reporting script.
- Initial comparison statistics, state-only diagnostics, punctuation replay,
  descriptive domain weighting, and a narrow clinical-note label-only baseline.
- An inference-free clinical-note label-identity report that verifies saved paired
  artifacts and compares complete predicted triples on the all-request denominator.
- The current all-eight-model Figure 2 renderer and focused offline tests.
- Instructions for binding newly generated local artifact hashes before a fresh
  rerun, with original paper hashes retained as provenance.

Shared modules supply parsing, validation, bootstrap calculations, request
construction, and local model loading. `target_spec.py` remains the earlier narrow
extraction of the target-loader and name recovery used by synthetic construction;
the unrelated P1/P2 implementations remain excluded. Follow-ups use their own
`python -m cpg2.<module>` entry points instead of importing the broad development CLI.

The retained `make_ml4h_findings_contract_figure_v0_7.py` script still computes
initial synthetic interactions and state-only reversal diagnostics used by the
current text. Its older plot is not the current manuscript Figure 1 or Figure 2.
The equal-domain analysis is supporting sensitivity code, not a new primary endpoint.

## Excluded

- E4 annotation/reviewer packets, repeated-review/time-gate workflows, retrieval
  expansion, MIMIC-IV workflows, and controlled-missingness experiments.
- Patient, admission, note, candidate, request, or subject-level MIMIC records,
  including pseudonymous row-level identifiers.
- Source clinical text, instantiated patient prompts, raw model responses,
  checkpoint/log files, and all working-repository aggregate result files.
- Synthetic source datasets and generated synthetic responses.
- Model weights, tokenizers, local snapshot directories, credentials, and caches.
- Manuscript source/PDFs, author-created conceptual artwork, submission templates,
  archived contexts, and internal evidence logs.
- Broad cross-task synthesis code requiring the excluded E4 artifacts.
- A new clinical adjudication, model inference run, or release of any data.

Only invented unit-test records and the pre-existing non-patient mechanical tree
are distributed as examples. No real dataset row was copied to make a test pass.

## Portability and provenance

Scientific fields (prompts, schema/validator rules, panel/revisions, seeds, budgets,
endpoints, and resampling definitions) are preserved. Repository-relative source
and model paths replace machine-specific locations. Config hashes referencing
those portable configurations are updated; source-file hashes still identify the
expected upstream inputs.

Dataset manifests and audit files record paths and/or local configuration hashes.
Their bytes therefore change on a fresh preparation. Use the documented binding
steps before running inference. Local bound configurations are ignored, remain
local, and must not be changed after execution starts. A mismatch in an existing
run remains an error, not an invitation to rewrite its provenance.

The exact inference-source hashes recorded for the author's original runs will
not match every portability adaptation here. Replaying those archived runs
requires their original hash-compatible workspace. This release provides code
for independent reruns; it does not claim an end-to-end rerun was performed while
preparing this release.

The compact runner's release correction counts tokenizer-added special tokens
in preflight and validates observed prompt lengths while preserving the original
string-input generation call. Its changed source hash requires a fresh run root;
old audits are not rewritten. The separate clinical-note label-identity report
can check archived paired outputs with their original configuration and recorded
generation provenance, without loading models or requiring matching installed
inference packages. Its current analysis-source hashes are recorded separately.

## Review and publication boundary

The entire `data/`, `models/`, `outputs/`, and `paper_artifacts/` trees are ignored.
Ignore rules reduce accidental inclusion but are not a privacy review and do not
protect files deliberately force-added. Review the complete changed-file list
and all untracked files before staging or publishing.

The owner selected the MIT License and identified UXFactory as the copyright
holder for original contributions. CPGPrompt-derived configurations retain the
upstream MIT notice; see [third-party notices](THIRD_PARTY_NOTICES.md). External
data and model terms are unchanged.

The owner authorized publication on 2026-09-29. The manuscript is not part of this
release and was not changed during publication. Its code-availability statement
should cite the verified public commit when it is updated.
