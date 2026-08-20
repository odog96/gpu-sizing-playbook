# Cloudera Blueprint: GPU Memory Sizing Playbook for Foundation Models

> Line-item memory accounting for training, fine-tuning, and inference of
> transformer models — with formulas validated against real H100 sweeps and
> companion spreadsheets that turn the math into sizing decisions.

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Use Case](#use-case)
- [Key Features](#key-features)
- [Quickstart](#quickstart)
- [Architecture / Software Components](#architecture--software-components)
- [Target Audience](#target-audience)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Documentation](#documentation)

## Overview

A three-part article series and its supporting artifacts — formulas,
spreadsheets, and a validation benchmark — that answer a practical question:
*given a model, a workload, and a GPU, will the job fit?* Article 1 covers
from-scratch training. Article 2 covers LoRA-style fine-tuning. Article 3
(inference) is in progress. Every numeric claim in the articles is validated
against a tracked H100 sweep that lives in this repo.

## Demo

_Reprise walkthrough — TBD._

## Use Case

Sizing GPU infrastructure for foundation-model workloads is usually done by
guess-and-check: pick a card, launch the job, watch it OOM, try a bigger
card. This playbook replaces that loop with a per-line-item memory budget
the reader can compute up front — for training, for LoRA fine-tuning, and
(soon) for inference — so hardware decisions and workload decisions are
answered from one accounting rather than from trial and error.

## Key Features

- Term-by-term memory formulas (weights, gradients, optimizer state,
  activations, KV cache) validated against measured GPU sweeps
- Companion spreadsheets — plug in model + config, get a fit/no-fit answer
  and per-line-item breakdown
- Reference H100 sweeps (24-config training, 25-config fine-tune) with worst
  non-OOM prediction error under 8.3%
- A benchmark harness that measures the same quantities the formulas
  predict, so drift between formula and reality stays visible

## Quickstart

```bash
# Install
pip install -r requirements.txt              # training
pip install -r requirements-finetune.txt     # fine-tune (adds transformers, peft)

# Read the articles
open articles/0-model-selection/article-0-model-selection.md   # right model, right job
open articles/1-training/article-1-training.md                  # training memory
open articles/2-finetuning/article-2-finetuning.md              # fine-tune memory

# Use the sizing tools
open assets/gpu-training-memory-sizing.xlsx
open assets/gpu-finetune-memory-sizing.xlsx

# Run the CPU-only validation suite (no GPU needed)
cd benchmark && python -m unittest discover -s tests

# Run the GPU sweep (see docs/benchmark-runbook.md for the full walkthrough)
cd benchmark && python benchmark.py --smoke-test              # <1 min sanity check
cd benchmark && python benchmark.py --output results.csv      # full training sweep
```

## Architecture / Software Components

Two halves that must stay in sync: `predictions.py` (what memory *should*
be, from formulas) and `train.py`/`memory_accounting.py` (what memory
*actually is*, measured on a running model). `benchmark.py` runs both for
the same config and puts them side by side in one CSV row. The reference
runs under `benchmark/reference-run/` and `benchmark/reference-run-finetune/`
are the tracked evidence behind the articles' numeric claims.

Full architecture notes: `docs/benchmark-manual.md`.

## Target Audience

- Platform administrators sizing GPU capacity for internal ML workloads
- ML engineers deciding which GPU tier a training or fine-tune run needs
- Data scientists translating a model choice into a hardware requirement
- Technical leaders evaluating cloud-GPU spend against workload footprint

## Repository Structure

| Path | Description |
| --- | --- |
| `articles/` | The article series (one folder per article, with markdown + figures) |
| `assets/` | Companion spreadsheets — the sizing tools |
| `benchmark/` | Formula code, measurement code, sweep orchestrator, tracked reference runs, CPU-only tests |
| `deploy/` | SSH-based remote-runner scripts for one-shot GPU sweeps |
| `docs/` | Benchmark manual (`benchmark-manual.md`) and runbook (`benchmark-runbook.md`) |
| `requirements*.txt` | Python dependencies for the benchmark |
| `METADATA.yaml` | Catalog metadata for the Cloudera blueprint website |

## Prerequisites

- Python 3.10+
- For the GPU sweep: a CUDA-capable GPU (H100 80 GB for the reference sweep;
  smaller cards work for smaller configs) and a working PyTorch CUDA build
- For the fine-tune sweep: the training prerequisites plus
  `transformers`/`peft` from `requirements-finetune.txt`
- No GPU is required to read the articles, use the spreadsheets, or run the
  CPU-only unit tests

## Hardware Requirements

| Deployment | Minimum |
| --- | --- |
| Read / spreadsheet-only | CPU workstation, no GPU |
| Reproduce reference sweep | 1 × H100 80 GB (or A100 80 GB with slightly reduced config coverage) |

## Documentation

- Articles: [`articles/`](articles/)
- Benchmark manual: [`docs/benchmark-manual.md`](docs/benchmark-manual.md)
- Benchmark runbook (exact GPU commands + expected OOMs): [`docs/benchmark-runbook.md`](docs/benchmark-runbook.md)
- Memory-formula references (per-article working docs):
  [`articles/1-training/memory-formulas.md`](articles/1-training/memory-formulas.md)
  and [`articles/2-finetuning/memory-formulas-finetune.md`](articles/2-finetuning/memory-formulas-finetune.md)
