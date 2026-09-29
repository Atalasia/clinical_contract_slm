# v0.13 code release: verification record

Prepared 2026-09-28; reproducibility fixes verified 2026-09-29.
Package candidate: **0.2.0rc1**.
Base commit: `2603548d056f8104f12e2b0e5d542585b2487d88`.

The owner authorized publication on 2026-09-29. This record describes the checks
performed before committing the release. The manuscript and its public-code
availability statement were not changed. No new model inference was run.

## What changed

| Addition | Why it belongs in this paper release |
|---|---|
| Six compact/paired-note/analysis modules and focused tests | Reproduce the matched experiments and their fixed scoring, normalization, and paired intervals |
| Compact prompt and two follow-up configurations | Declare the actual interventions, runtime settings, counts, and original hash provenance |
| Compact report and saved-output sensitivity scripts | Reproduce Table I and the label/abstention/record decomposition |
| Current all-model Figure 2 renderer | Plot stored synthetic and note contrasts without recomputing intervals |
| Isolated clinical-note baseline report | Reproduce the constant-label comparison without pulling in the old E4-dependent cross-task pipeline |
| Domain-weighting sensitivity script | Preserve the associated descriptive robustness analysis |
| Local artifact-binding utility | Make fresh portable reruns possible without disabling provenance checks or relabeling old results |
| Updated requirements and reproduction guide | Pin the follow-up environment, retain legacy plotting pins separately, and map paper results to commands |
| MIT license and third-party notices | Record the owner's license choice and UXFactory copyright; preserve CPGPrompt's upstream MIT notice and distinguish external data/model terms |
| Corrected compact prompt-token accounting | Match preflight to the historical string-input special-token behavior, and reject runtime/resume count mismatches without changing generation |
| Saved clinical-note label-identity report | Reproduce the cross-condition triple comparison from paired raw responses, with all-request denominators and archived provenance preserved |

The original narrow CLI and portable shared modules remain intact. Follow-up
runners use their existing `python -m` entry points. Model revisions, prompts,
validators, endpoints, sampling budgets, and statistical definitions were not
retuned for this release. The saved-output analyzer gained an optional explicit
configuration path so it can verify a separately bound local run. The compact
runner's later token-accounting correction preserves the generation call but
changes source/preflight provenance; earlier cells must not be relabeled or
resumed with it. See the reproduction guide for fresh-root and archived-output
instructions.

## What was verified

- Final offline suite: **212 tests run; 210 passed and 2 expected integration
  skips**. Tests cover adapters, scorers, resampling, mocked runners,
  checkpoint/resumption checks, local bindings, figure validation and baseline
  provenance.
- Both follow-up runner help commands, the original CLI, and all seven new script
  help commands execute without loading models.
- Python/JSON/TOML syntax checks pass; package optional dependencies match all
  three requirements files.
- Portable cross-configuration, prompt and wrapper hash tests pass.
- Four unchanged inference/scoring follow-up modules and the compact prompt
  match their development originals byte-for-byte. The compact runner differs
  only for prompt-token accounting and validation; its engine setup, model
  contexts, generation call, and sampling calls match the original implementation.
  The two new experiment configurations differ only in portable primary-config
  hashes and explicit release provenance.
- Token-accounting regressions cover an extra beginning-of-sequence token at
  the 2,048-token boundary, unchanged generation arguments, runtime mismatches,
  and stale/completed-cell validation. Label-identity regressions cover shuffled
  pairing, equal accuracy with different predicted triples, invalid-output
  denominators, conservative fence removal, literal abstention/native labels,
  archived provenance, artifact checksums, and private non-overwriting reports.
  These tests use invented fixtures. A separate read-only check of the author's
  archived paired responses reproduced **2,257/2,288 identical label triples
  (98.6%)**. The primary comparison uses schema-valid literal labels, regardless
  of logical consistency; the stricter secondary comparison is 2,255/2,286 among
  pairs that are logically valid in both arms. Only aggregate counts were
  displayed; no raw responses or patient-level records were exported or bundled.
- The release Figure 2 `--check-only` command validates all eight models and
  32 saved estimate/95% interval triplets against the author's two existing
  aggregate files, including denominators and contrast orientations. No aggregate
  was copied into the release and no raw response or patient-level file was read
  for that check.
- Review of all 111 candidate files and targeted path/credential scans found no included data,
  outputs, weights, manuscript artifacts, private keys or developer absolute
  paths. Tracked/untracked source inventory contains no data symlinks.
- `git diff --check` passes. The public remote was checked against the base
  commit before publication; no intervening remote changes were found.

These checks do not establish an end-to-end GPU rerun or identical numerical
results on another machine. Two integration tests require intentionally excluded
inputs and are expected to skip: upstream CPGPrompt source files and the locally
retained frozen synthetic dataset manifest. Test fixtures elsewhere are invented.

## How to inspect the release

From this directory:

```bash
git status --short
git diff --stat 2603548 HEAD
git diff 2603548 HEAD -- README.md RELEASE_SCOPE.md pyproject.toml tests/test_release_config_hashes.py
git ls-files
```

The comparison uses the original public release as its base. Start with
[the reproduction guide](docs/REPRODUCING_V13.md) and
[the scope boundary](RELEASE_SCOPE.md).

To repeat offline tests with a configured environment:

```bash
PYTHONPATH=src python -m unittest discover -s tests -t . -v
```

## Publication boundary

The owner selected MIT, confirmed UXFactory as copyright holder, and authorized
the updated code push on 2026-09-29. CPGPrompt's upstream MIT notice is retained in
`LICENSES/CPGPrompt-MIT.txt`, with affected files listed in
`THIRD_PARTY_NOTICES.md`. Package metadata includes all three licensing files.
Local datasets and outputs are excluded from publication.

Use the containing release commit as the immutable code reference after verifying
it on the public remote. Updating the manuscript's code-availability statement is
a separate step; this release does not change the manuscript.
