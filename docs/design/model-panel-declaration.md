# Sub-5B SLM model-panel declaration: portable record

Status: approved and frozen on 2026-08-28

Machine-readable declaration: `configs/slm_benchmark/benchmark.v2.json`

This portable record documents the identities and interpretation of the released panel.
The eight-model set is a six-model mechanistic core plus two descriptive references; it
is not a balanced eight-model factorial design.

| Manifest ID | Released model | Immutable revision | Role |
|---|---|---|---|
| `q15_pt_v1` | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | Core: small base |
| `q15_it_v1` | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Core: small Instruct |
| `q3_pt_v1` | `Qwen/Qwen2.5-3B` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | Core: medium base |
| `q3_it_v1` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Core: medium Instruct |
| `g4_it_v1` | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | Core: general Gemma package |
| `mg4_it_v1` | `google/medgemma-4b-it` | `290cda5eeccbee130f987c4ad74a59ae6f196408` | Core: medical Gemma package |
| `mg15_it_v1` | `google/medgemma-1.5-4b-it` | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` | Descriptive medical reference |
| `q35_ref_v1` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Descriptive general reference |

The five core contrast identities are:

1. Qwen2.5 1.5B Instruct minus base;
2. Qwen2.5 3B Instruct minus base;
3. Qwen2.5 base 3B minus 1.5B;
4. Qwen2.5 Instruct 3B minus 1.5B; and
5. MedGemma 4B Instruct minus Gemma 3 4B Instruct.

Two additional contrasts compare MedGemma release generations and Qwen3.5 4B with
Qwen2.5 3B Instruct. They are descriptive because architecture, data, post-training,
release age, or multiple package components can differ together.

All eight acquired models were retained after technical compatibility evaluation.
Model-specific clinical hints were prohibited. The exact license identifiers, parameter
metadata, local snapshot locations, wrapper specifications, and verification hashes are
recorded in `configs/slm_benchmark/models/`; model files themselves are not distributed.
