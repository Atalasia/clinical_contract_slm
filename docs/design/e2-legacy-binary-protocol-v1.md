# Historical forced-binary tree protocol: portable record

Status: frozen E2-A protocol with post-hoc punctuation-policy extension

Machine-readable contract: `configs/slm_benchmark/e2.historical_tree_binary.v1.json`

E2-A traverses the historical CPGPrompt decision trees for all source vignettes. At
each visited query, the model receives the frozen binary prompt and one output is
retained. Runtime failures remain failures under every policy.

The prespecified strict evaluator trims surrounding whitespace and accepts only a
complete, case-insensitive `Yes` or `No`. A nonconforming output stops that path and the
case remains incorrect. The prespecified legacy sensitivity extracts the first
standalone binary token and defaults a tokenless response to `No`, then continues the
adaptive traversal.

The Findings paper additionally reports a post-hoc, inference-free replay that accepts
a complete binary response with one optional terminal period. It performs no substring
extraction, Markdown repair, or default-to-`No` coercion. The versioned implementation
is `scripts/analyze_e2_punctuation_policy_v0_7.py`.

The endpoint is the equal-weight mean of terminal-action exact accuracy across
headache, low back pain, and prostate cancer. Intervals use 10,000 paired vignette
resamples within domain, shared across models and policies, with seed 2026 and empirical
percentile 95% limits. Invalid paths remain in the denominator. The per-domain majority
terminal is an in-sample descriptive comparator; all-`No` is a deterministic traversal
comparator.

This historical task shares its source vignettes with synthetic criterion grounding.
Its reference paths were reconstructed from source metadata and were not independently
clinician adjudicated. Results measure compatibility with the historical tree task and
evaluator behavior, not clinical validity or prospective performance.
