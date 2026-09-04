# Output contracts in clinical SLM evaluation

This directory is the deliberately narrow code release for the accompanying paper. It contains the code and frozen non-patient configuration needed for:

- E1 synthetic typed-state criterion grounding;
- E1-S2 JSON-schema-constrained decoding sensitivity;
- E2 historical CPGPrompt tree traversal and evaluator-policy sensitivity;
- E3 MIMIC-III-Ext-Notes candidate contextualization; and
- the paper's clustered interaction analysis and main contrast figure.

It is not a copy of the full development repository. Earlier MIMIC-IV experiments, E4 annotation workflows, exploratory retrieval pipelines, manuscript drafts, model weights, source datasets, prompts instantiated with patient text, raw responses, and patient-level artifacts are intentionally absent.

## Repository layout

```text
configs/        Portable copies of the task, model, prompt, wrapper, tree, and registry specifications
examples/       Non-patient mechanical tree used by unit tests
scripts/        Paper interaction analysis and Figure 1 generator
src/cpg2/       Minimal execution and evaluation package plus dependency closure
tests/          Focused unit tests for the released code
```

`RELEASE_SCOPE.md` lists the inclusion and exclusion boundary in more detail.

### Configuration provenance

The scientific settings, immutable model revisions, prompts, wrappers, source-content hashes, endpoints, and seeds are unchanged from the paper run. Machine-specific model, CPGPrompt, and MIMIC paths were replaced with ignored repository-local locations. Because path strings contribute to the E1 configuration hash, the E1-S2 reference to that configuration was updated to the portable copy's hash. Hashes of the original paper-run dataset manifest and aggregate result remain preserved as provenance fields.

## Environment

The recorded inference environment used Python 3.12.3, vLLM 0.22.1, Transformers 5.10.2, PyTorch 2.11.0, Accelerate 1.13.0, and huggingface-hub 1.18.0. The complete local versions used by the release are pinned in `requirements-inference.txt`. CUDA-compatible PyTorch and vLLM installation can depend on the host platform, so install the appropriate wheels for the target CUDA stack when a direct `pip` install is unsuitable.

For the standard-library unit tests:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -v
```

Install the pinned Figure 1 analysis environment separately with:

```bash
python -m pip install -r requirements-analysis.txt
```

For model inference, install the inference requirements or the equivalent CUDA-specific environment:

```bash
python -m pip install -r requirements-inference.txt
```

## Inputs that are intentionally not included

Run commands from this repository's root. Supply the source material at these locations:

```text
data/cpgprompt/headache/vignettes_data.csv
data/cpgprompt/headache/merged_decision_tree.json
data/cpgprompt/lower_back_pain/lower_back_pain_results_short.csv
data/cpgprompt/lower_back_pain/decision_tree_lower_back_pain.json
data/cpgprompt/prostate_cancer/vignettes_data_pc_test.csv
data/cpgprompt/prostate_cancer/decision_trees_output_text.json
data/mimic-iii-ext-notes-1.0.0/notes.csv
data/mimic-iii-ext-notes-1.0.0/labels.csv
```

The CPGPrompt files must be obtained from the upstream project. MIMIC-III-Ext-Notes must be obtained through PhysioNet under its credentialing and data-use requirements. The frozen configurations verify the expected source hashes before constructing requests.

The portable E1 constructor should produce a request JSONL with SHA-256 `0887f1d3c17464134d0bb37f8b61b82b080eefa260752c2bf191a88180219543`. Its manifest byte hash will differ from the original working artifact because the manifest records local configuration and source paths. Consequently, a from-scratch rerun must update `dataset_manifest_sha256` and `primary_results_sha256` in `e1_s2.structured_output.v1.json` after locally regenerating and verifying those two upstream artifacts. The preserved values identify the paper run; they are not a substitute for checking a new local run.

Place or symlink each exact immutable model revision at the relative `model.local_snapshot` location declared in `configs/slm_benchmark/models/` (for example, `models/q15_pt_v1`). Model files are ignored by Git and must never be added to this repository.

## Reproduction order

The command-line interface exposes only the paper-relevant stages:

```bash
cpg2-paper --help
cpg2-paper slm-benchmark-e1-prepare --help
cpg2-paper slm-benchmark-e1-s2-run --help
cpg2-paper slm-benchmark-e2-run --help
cpg2-paper slm-benchmark-e3-run --help
```

Run the stages in this order:

1. populate the immutable model snapshot paths;
2. validate the benchmark configuration;
3. prepare, token-audit, infer, and aggregate E1;
4. token-audit, infer, and aggregate E1-S2 on the same E1 manifest;
5. prepare, token-audit, infer, and aggregate E2;
6. run `scripts/analyze_e2_punctuation_policy_v0_7.py` to reproduce the post-hoc, inference-free punctuation-policy diagnostic from the frozen E2 decision logs;
7. locally audit MIMIC-III-Ext-Notes, then prepare, infer, and aggregate E3; and
8. run `scripts/make_ml4h_findings_contract_figure_v0_7.py` after the E1 and E1-S2 response trees occupy the default `outputs/local_restricted/slm_benchmark_v2/` layout.

The Figure 1 script accepts path overrides through `CPG2_E1_RESULTS`, `CPG2_E1_S2_RESULTS`, `CPG2_E1_RESPONSES`, `CPG2_E1_S2_RESPONSES`, `CPG2_FIGURE_OUTPUT_DIR`, and `CPG2_DERIVED_OUTPUT`. The punctuation replay accepts `--e2-root` and `--output`. These options allow analysis of locally retained outputs without copying them into the repository.

All request and response JSONL files are written with restricted permissions by the execution code. Keep the complete `outputs/` tree local. Only independently reviewed, privacy-safe aggregate material should ever be considered for release.

## Important boundaries

- E1 and E2 source vignettes are not blinded holdouts.
- E1-S2 is an exploratory output-contract sensitivity analysis.
- The punctuation-normalized E2 policy is a post-hoc, inference-free diagnostic.
- E3 evaluates released MetaMap candidates in single nursing notes; it is not end-to-end extraction or longitudinal state inference.
- The code measures agreement, format compliance, and evaluator behavior. It does not establish clinical safety or patient benefit.
- No software license has been selected in this staging copy. Add an explicit license before a public release if reuse is intended.
