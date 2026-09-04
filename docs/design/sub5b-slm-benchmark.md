# Sub-5B clinical SLM benchmark: portable design record

Status: frozen paper-release provenance record

Machine-readable contract: `configs/slm_benchmark/benchmark.v2.json`

This file is the portable release counterpart of the study's longer internal design
record. It preserves the design facts needed to interpret and validate the released
configuration without importing unrelated internal project history.

## Study question and scope

The benchmark asks whether output-contract choices can change selection among locally
served language models with no more than approximately five billion language-model
parameters. The output contract includes the prompt or template, decoding constraint,
parser, and failure policy. The study compares released packages and does not identify
causal effects of model scale, instruction tuning, or medical adaptation.

The released study components are:

1. typed-state grounding of criterion evidence in synthetic CPGPrompt vignettes;
2. an exploratory JSON-schema-constrained rerun of the identical synthetic requests;
3. historical forced-binary CPGPrompt tree traversal with strict and legacy evaluator
   policies, plus a post-hoc punctuation-only replay; and
4. candidate-conditioned contextualization in MIMIC-III-Ext-Notes.

Synthetic state-macro agreement was adopted post hoc for the Findings paper as the
balanced model-selection endpoint. Strict micro agreement is the request-weighted
operational endpoint. The real-note task uses its native hierarchical joint endpoint,
and tree traversal uses the equal-weight mean of terminal-action accuracy across three
clinical domains. Scores from different tasks are not averaged into a composite.

## Frozen comparison structure

The panel contains a six-model mechanistic core and two descriptive contemporary
references. Five core contrast identities cover Qwen2.5 instruction status at two
sizes, Qwen2.5 size within two instruction states, and the released MedGemma 4B versus
Gemma 3 4B package. MedGemma 1.5 4B and Qwen3.5 4B comparisons are descriptive. Exact
model identifiers, immutable revisions, roles, and wrapper hashes are recorded in the
benchmark configuration and its model manifests.

All generation used local BF16 vLLM inference without quantization, temperature zero,
seed 2026, and frozen task-specific output allowances. Base checkpoints used the
declared plain-completion wrapper; other releases used their declared official tokenizer
or processor template. Invalid output remained an error under strict scoring.

## Analysis and claim boundaries

The constrained synthetic condition was designed after inspection of the unconstrained
results and is exploratory. Bootstrap intervals quantify vignette-resampling stability
within the evaluated benchmark. The historical tree analysis shares source vignettes
with synthetic grounding. MIMIC-III-Ext-Notes is a distinct task and provides contextual
ranking evidence, not a direct transfer estimate. None of these tasks establishes
clinical safety, calibration, patient benefit, or prospective generalization.

Source datasets, model weights, raw responses, clinical text, and patient-level
artifacts are intentionally excluded from this repository. `README.md` and
`RELEASE_SCOPE.md` describe how authorized users supply those inputs and the exact
release boundary.
