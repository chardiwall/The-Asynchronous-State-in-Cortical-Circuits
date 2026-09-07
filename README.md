# The Asynchronous State — reproduction and extension

Reproducing Renart et al. 2010, *The Asynchronous State in Cortical Circuits*
(*Science* **327**, 587), then replacing its network and its input to test whether the
asynchronous state survives.

## The four levels

| Level | Network | Input | Question |
|-------|---------|-------|----------|
| **L1** | paper's (binary + conductance-based LIF) | paper's (Poisson) | Can we reproduce it, and do we understand it? |
| **L2** | **spiking reservoir (LSM)** | paper's (Poisson) | Does the asynchronous state hold with a reservoir? |
| **L3** | paper's | **N-MNIST, DVS-Gesture** | Does it hold under structured event-based input? |
| **L4** | **spiking reservoir (LSM)** | **N-MNIST, DVS-Gesture** | Does it hold when both are replaced? |

The measurement is the invariant across all four: population-averaged spiking correlation `r̄`,
the width `σ_r` of its distribution, the current-component correlations
`c_EE / c_II / c_EI / c`, and how they scale with network size `N`.

## Where things are

| Path | Contents |
|------|----------|
| [`paper.md`](paper.md) | index and reading map for the paper — **start here** |
| `docs/paper/` | full implementation-level extraction of the paper, split by block |
| `docs/adr/` | architecture decision records (one file per decision) |
| `docs/*.pdf` | the source PDFs |
| `config.yaml` | every seed, path and model parameter — nothing is hardcoded in code |
| `src/` | `dataset.py`, `model.py`, `train.py`, `eval.py` |
| `tests/` | tests, written before the code they cover |
| `data/raw/` | **read-only** source data |
| `data/processed/` | generated data |
| `artifacts/` | metrics, logs, figures |
| [`PROGRESS.md`](PROGRESS.md) | current session's objectives and completion criteria |
| [`CONTEXT.md`](CONTEXT.md) | glossary of project terms |
| [`reviews.md`](reviews.md) | human-readable session log (written by `/finalise`) |
| [`AGENTS.md`](AGENTS.md) | agent-facing session log (written by `/finalise`) |

## Status

Documentation phase complete; no simulation code written yet. Simulation framework decided:
**Brian2** ([ADR 0001](docs/adr/0001-brian2-as-simulation-framework.md)). Next up is the shared
analysis pipeline (test-first), then Fig. 3A–B as the first reproduction target.

## Working agreement

See [`CLAUDE.md`](CLAUDE.md). In short: objectives and completion criteria are agreed before
work starts; questions are asked one at a time; mathematics is taken from a library where one
exists and reviewed line by line where it doesn't; tests come before implementation; files stay
under 200 lines; and nothing is called "done" until it has run and written metrics to a log.
