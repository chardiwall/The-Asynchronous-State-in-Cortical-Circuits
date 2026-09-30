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
| `src/analysis.py` | the shared measurement pipeline, used identically by every block |
| `src/block1/` | Fig. 1 — feedforward postsynaptic pair ([README](src/block1/README.md)) |
| `src/block2/` | Fig. 2 — recurrent binary network ([README](src/block2/README.md)) |
| `src/block3/` | Fig. 3 — recurrent conductance-based spiking network ([README](src/block3/README.md)) |
| `tests/` | tests, written before the code they cover |
| `data/raw/` | **read-only** source data |
| `data/processed/` | generated data |
| `artifacts/` | metrics, logs, figures |
| [`PROGRESS.md`](PROGRESS.md) | current session's objectives and completion criteria |
| [`CONTEXT.md`](CONTEXT.md) | glossary of project terms |
| [`reviews.md`](reviews.md) | human-readable session log (written by `/finalise`) |
| [`AGENTS.md`](AGENTS.md) | agent-facing session log (written by `/finalise`) |

## Status

All three L1 blocks are implemented for the paper's **main-text** figures. Each block's own
README is the run guide: exact commands per panel, which `config.yaml` keys trade cost against
accuracy, the paper's target result, and that block's known deviations from the paper.

| Block | Figure | Code | Results |
|---|---|---|---|
| 1 | Fig. 1 B/C/E/F | complete | produced |
| 2 | Fig. 2 B/C/D/E/G | complete | Fig. 2C's two sweeps are long cluster jobs; see `PROGRESS.md` |
| 3 | Fig. 3 A/B/C/D | complete | Fig. 3A–B partially produced; Fig. 3C–D not yet run |

Fig. 2A and 2F are schematics in the paper, not simulations, so nothing is generated for them.
Supplementary Figs. S2–S8 are **out of scope** so far.

`PROGRESS.md` is authoritative for what has actually been run, at what scale, with what
numbers. Two values in Fig. 3C–D's configuration are inferences rather than transcriptions and
are flagged at the top of [`src/block3/README.md`](src/block3/README.md).

## Working agreement

See [`CLAUDE.md`](CLAUDE.md). In short: objectives and completion criteria are agreed before
work starts; questions are asked one at a time; mathematics is taken from a library where one
exists and reviewed line by line where it doesn't; tests come before implementation; files stay
under 200 lines; and nothing is called "done" until it has run and written metrics to a log.
