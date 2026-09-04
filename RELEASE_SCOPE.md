# Release scope

## Included

- benchmark, model-revision, prompt, wrapper, and task configuration;
- typed-state output parsing and validation;
- deterministic synthetic request construction;
- local vLLM inference for E1, E1-S2, E2, and E3;
- strict and legacy evaluator policies plus the post-hoc punctuation-normalized replay used in the paper;
- model-level aggregation, paired comparisons, clustered bootstrap procedures, and baselines;
- the direct contract-by-model interaction analysis and Figure 1 renderer; and
- focused unit tests for the included implementation.

`target_spec.py` contains only the target-loader and legacy-name recovery used by E1 construction. These mechanisms were extracted without changing their validation logic so the older P1/P2 and retrieval pipelines did not have to be included. Unrelated command-line entry points were also removed.

## Excluded

- all MIMIC-IV, E4, reviewer-packet, time-gate, retrieval-expansion, and controlled-missingness code;
- all raw or processed clinical text;
- all patient-, admission-, note-, candidate-, or request-level MIMIC artifacts;
- all model outputs, including synthetic outputs;
- all aggregate result files from the working repository;
- model weights, tokenizer caches, and local snapshot directories;
- manuscript source, internal design notes, evidence logs, and archived project context; and
- generic experiments that do not contribute to a result reported in the paper.

The absence of output files is intentional. The `.gitignore` prevents the standard local data, model, output, and generated-paper-artifact locations from being staged accidentally.
