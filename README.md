# IntentBench

Parameter-efficient fine-tuning and benchmarking of a small open-source
instruction-tuned LLM for intent classification.

> **Benchmark results go here.** Run the pipeline below and copy the numbers
> from `outputs/metrics/benchmark.json` into this table — do not report
> placeholder figures.

| Run | Accuracy | Macro-F1 | Weighted-F1 | Invalid-output rate | P50 / P95 latency (ms) | Peak GPU mem (GB) |
|---|---|---|---|---|---|---|
| Majority-class | — | — | — | — | — | — |
| Zero-shot | — | — | — | — | — | — |
| Few-shot | — | — | — | — | — | — |
| QLoRA fine-tuned | — | — | — | — | — | — |

Full report: [`docs/benchmark_report.md`](docs/benchmark_report.md).

## What this is

- **Base model:** `google/gemma-2-2b-it` (instruction-tuned, 4-bit NF4 quantized)
- **Fine-tuning:** QLoRA (Hugging Face Transformers + PEFT + bitsandbytes), one
  defensible default configuration, trained once
- **Dataset:** [Banking77](https://huggingface.co/datasets/banking77) —
  77-class single-label customer-intent classification, CC BY 4.0
- **Baselines:** majority-class, zero-shot, few-shot (fixed example set) —
  each defined once and reused, so the fine-tuning gain is measured against
  a stable floor
- **Evaluation:** macro-F1 (primary), accuracy, macro/weighted precision &
  recall, per-class P/R, invalid-output rate, confusion matrix, top-20
  confusion pairs, inference latency (P50/P95), peak GPU memory
- **Serving:** FastAPI `/classify` + `/health` + `/model-info`, model loaded
  once at startup

This is a scoped, 4-day implementation. Rigor is preserved (frozen test set,
real baselines, real metrics); breadth is cut — see [Limitations & future
work](#limitations--future-work).

## Repository structure

```
intentbench/
├── README.md
├── requirements.txt        # pinned versions -- replaces Dockerfile's role
├── run_api.sh              # local serving entrypoint -- replaces container CMD
├── .env.example
├── configs/
│   ├── base.yaml            # data/model/eval settings shared by every script
│   └── qlora.yaml           # QLoRA hyperparameters for the one training run
├── src/
│   ├── data.py               # load, profile, split Banking77
│   ├── prompts.py            # prompt construction + output parsing
│   ├── model.py               # quantized model + LoRA adapter loading
│   ├── train.py                # QLoRA training entrypoint
│   ├── evaluate.py             # unified benchmark script
│   ├── metrics.py               # metric computation (pure functions, unit-tested)
│   ├── plots.py                  # confusion matrix plotting
│   └── utils.py                   # seeding, config loading
├── baseline/
│   ├── majority_class.py
│   ├── zero_shot.py
│   └── few_shot.py
├── serving/
│   ├── app.py               # FastAPI routes
│   ├── model_service.py      # loads model once, mockable in tests
│   └── schemas.py             # Pydantic request/response models
├── tests/
│   ├── test_metrics.py
│   ├── test_prompts.py
│   └── test_api.py           # API tests against a mocked model service
├── outputs/
│   ├── metrics/
│   ├── plots/
│   └── predictions/
└── docs/
    └── benchmark_report.md
```

No `Dockerfile` — Docker isn't available in this environment. See
[Reproducibility without Docker](#reproducibility-without-docker).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or the pip install line below, then freeze

# 1. Data profiling + frozen split (Day 1)
python -c "from src.data import build_data_profile; from src.utils import load_config; build_data_profile(load_config('configs/base.yaml'), 'outputs/metrics/data_profile.json')"

# 2. Baselines (Day 1)
python -m baseline.majority_class
python -m baseline.zero_shot
python -m baseline.few_shot

# 3. QLoRA fine-tuning (Day 2) -- requires a CUDA GPU (e.g. Colab T4)
python -m src.train

# 4. Evaluation on the frozen test set, run exactly once (Day 3)
python -m src.evaluate --adapter-path outputs/best --run-id qlora_r16_lr2e-4_seed42

# 5. Serve the fine-tuned model (Day 4)
./run_api.sh
```

Environment setup mirrors the build plan:

```bash
pip install -U torch transformers datasets peft bitsandbytes accelerate \
  evaluate scikit-learn wandb fastapi uvicorn pydantic matplotlib seaborn pandas pytest
pip freeze > requirements.txt   # pin immediately after the first working install
```

## Running tests

Every deterministic component (metrics, prompt formatting, output parsing,
API contract) is unit-tested without a GPU or a real model load — the model
service is mocked (see `tests/test_api.py`):

```bash
pytest tests/ -v
```

A single live GPU smoke test that exercises the real model path is left as
a manual/CI step once a GPU environment is available, per the plan's
"mock the model, don't load it per test" approach.

## API contract

| Route | Input | Purpose |
|---|---|---|
| `GET /health` | none | status + loaded flag |
| `GET /model-info` | none | base model, quantization, adapter, dataset metadata |
| `POST /classify` | `{"text": "..."}` | intent + latency + model ID |

```json
POST /classify
{"text": "Why has my card payment been reversed?"}

200 OK
{"intent": "reverted_card_payment", "latency_ms": 42.1, "model": "intentbench-gemma-qlora"}
```

There is deliberately **no `confidence` field**: a generative model
constrained to one of 77 labels doesn't have a clean softmax-over-classes
the way a dedicated classification head does. Adding a plausible-looking
number with no defined scoring method would be fabricated, so it's omitted
until a real scoring method (e.g. token-level log-probability of the
generated label) is implemented and documented.

## Reproducibility without Docker

Docker isn't available in this environment. The same guarantees a
Dockerfile would encode are recorded here as data and scripts instead:

| What Docker would guarantee | How this project guarantees it |
|---|---|
| Exact package versions | `pip freeze > requirements.txt` immediately after the first working install, committed |
| GPU / driver / CUDA version | Recorded here: _fill in, e.g. Colab T4, `torch.version.cuda`_ |
| Random seeds | Fixed and logged for data split, training, and evaluation (`seed: 42` in `configs/base.yaml`) |
| Run configuration | Every training/eval run reads from a versioned `configs/*.yaml`, never hard-coded values |
| Serving entrypoint | `run_api.sh` — activates the venv and starts uvicorn |
| Label mapping | `labels.json` saved once during training, loaded (never redefined) by the serving code |

## Limitations & future work

Cut from the full IntentBench blueprint to fit a 4-day scope — not because
they're low-value, but because they aren't needed to prove the core skill
once:

- **LoRA rank / learning-rate / target-module ablations** — one defensible
  default configuration was trained and evaluated rigorously instead of a
  sweep.
- **A second model family** — only Gemma-2-2B-IT was fine-tuned; no
  cross-family comparison.
- **Out-of-scope (OOS) intent detection** — stretch goal, not required for
  MVP credibility.
- **Containerized deployment** — no Docker in this environment; see
  [Reproducibility without Docker](#reproducibility-without-docker) for the
  equivalent guarantees.
- **CI (GitHub Actions)** — polish, not signal for this scope; the test
  suite (`pytest tests/`) is fast and GPU-free enough to add later.
- **Hugging Face Hub publishing** — optional follow-up.

## Resume bullet

> Built IntentBench, a reproducible benchmark for parameter-efficient
> fine-tuning of a small open-source instruction-tuned LLM (Gemma-2-2B-IT)
> on Banking77 (77-class intent classification); fine-tuned with 4-bit QLoRA
> using Hugging Face Transformers/PEFT/bitsandbytes, and compared zero-shot,
> few-shot, and fine-tuned variants on macro-F1, macro precision/recall,
> per-class error, GPU memory, and inference latency; served via FastAPI.

This is a scoped implementation, not a claim about final benchmark results —
report the numbers your actual run produces on your actual hardware.
