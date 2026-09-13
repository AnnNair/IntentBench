# IntentBench Benchmark Report

> Fill this in from `outputs/metrics/benchmark.json` after the single Day-3
> evaluation run on the frozen Banking77 test set. Do not report numbers from
> anywhere else, and do not run the test set more than once.

## Setup

- Base model: `google/gemma-2-2b-it` (4-bit NF4)
- Fine-tuning: QLoRA, `configs/qlora.yaml`
- Dataset: Banking77 (77-class intent classification)
- Hardware: _fill in (e.g. Colab T4, CUDA x.y)_
- Seed: 42

## Results

| Run | Accuracy | Macro-F1 | Weighted-F1 | Invalid-output rate | P50 latency (ms) | P95 latency (ms) | Peak GPU mem (GB) |
|---|---|---|---|---|---|---|---|
| Majority-class | | | | | | | |
| Zero-shot | | | | | | | |
| Few-shot | | | | | | | |
| QLoRA (r=16, lr=2e-4) | | | | | | | |

## Error analysis

- Top confusion pairs: see `outputs/metrics/top_confusion_pairs.json`
- Full error list: see `outputs/predictions/errors.csv`
- Confusion matrix: see `outputs/plots/confusion_matrix.png`

## Limitations

- Single QLoRA configuration; no rank/LR/target-module ablation sweep (future work).
- Single model family (Gemma-2-2B); no cross-family comparison.
- No out-of-scope (OOS) intent detection.
- No containerized deployment; reproducibility relies on pinned
  `requirements.txt`, fixed seeds, and versioned YAML configs instead (see
  README.md, "Reproducibility without Docker").
