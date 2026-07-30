# Evaluation

Two fixtures, used for two different purposes that must not be mixed.

| Fixture | Purpose | How often you may run it |
|---|---|---|
| `calibration_fixture.yaml` | Choose chunk sizes, abstention thresholds, embedding dimension, and models | As often as you like |
| `holdout_v1.yaml` | Decide pass/fail on a frozen configuration | **Exactly once** |

## Why the separation is strict

Tuning a threshold until a test set passes means the score on that set is no
longer an estimate of real performance — it is a measure of how hard you tuned.
The holdout exists to give one honest number, which requires that the
configuration was fixed *before* the holdout was ever run.

## The lifecycle

1. Tune freely against `calibration_fixture.yaml`.
2. **Freeze** the configuration. Record its `config_hash` (printed by `run_eval.py`).
3. Run the holdout **once**.
4. **Pass** → the quality gate is met. Record the report.
5. **Fail** → `holdout_v1.yaml` is now **retired**. Its cases may be folded into
   the calibration fixture, and the next final evaluation needs a newly written,
   previously unseen `holdout_v2.yaml`.

A failed holdout never justifies relaxing a threshold and re-running the same
holdout. That converts the honest number into a tuned one, silently.

## Case kinds

| Kind | Answerable | What it tests |
|---|---|---|
| `direct` | yes | The fact is stated plainly; basic retrieval |
| `paraphrase` | yes | Retrieval without lexical overlap — the real test of embeddings |
| `later-page` | yes | Evidence past the first page or two. **The decisive regression test** for the pre-audit single-vector implementation, which embedded only the truncated head of a document |
| `cross-page` | yes | Requires combining facts from two separate pages |
| `misleading` | **no** | Shares vocabulary with the document but is not answered by it. The hardest abstention case, because retrieval returns confident-looking chunks |
| `unanswerable` | **no** | Plainly outside the document |

## Metrics

- **retrieval recall@k** — did the retrieved set contain the evidence at all
- **citation correctness** — did the cited page actually contain the answer
- **abstention precision / recall** — of the answers we refused, how many should
  have been refused, and of those we should have refused, how many were
- **unsupported-answer rate** — answered confidently when it should have abstained.
  The number that matters most for a public demo, because a fabricated answer is
  worse than no answer
- **latency** — median and p95 for `/ask`

## Running

```bash
# Calibration (tune freely)
python eval/run_eval.py --fixture eval/calibration_fixture.yaml

# Sweep a threshold to pick a value
python eval/run_eval.py --fixture eval/calibration_fixture.yaml \
    --sweep RERANK_MIN_SCORE=0.05,0.10,0.20,0.30,0.40

# Holdout: once, against the frozen configuration
python eval/run_eval.py --fixture eval/holdout_v1.yaml --holdout
```

Reports land in `eval/reports/<fixture>_<config_hash>.json` and are gitignored:
they embed question text and are regenerated on every run.

Every run needs real credentials and spends real Cohere tokens. The suite in
`tests/` covers correctness with fakes and no network; this measures quality.
