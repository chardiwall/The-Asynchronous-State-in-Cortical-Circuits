# Block 1 — Postsynaptic Pair (Fig. 1 reproduction)

Reproduces Block 1 of Renart et al. 2010, *The Asynchronous State in Cortical Circuits*
(*Science* 327): the feedforward postsynaptic-pair model behind main-text Fig. 1. This is
the `L1` level of this project (see the repo root `README.md`'s four-level table) — the
first, smallest reproduction target before Blocks 2/3 show the same mechanism holds in
full recurrent networks.

## The network

Two independent, current-based leaky-integrate-and-fire (LIF) neurons ("cell A", "cell B"),
each driven by its own feedforward presynaptic input. **There is no recurrent connectivity
between them** — this is purely feedforward, isolating how shared/correlated *input alone*
shapes *output* correlation, before the rest of the paper adds recurrence.

Membrane equation (S-p.19):
```
τ_m dV/dt = -V + J_E·Σs_i^E(t) - J_I·Σs_i^I(t)     (if V < θ)
```
On `V ≥ θ`: spike, reset to `V_R`, clamp for the refractory period `t_ref`. `J_E`, `J_I` are
calibrated (`calibration.py`) so a single presynaptic spike produces a ±0.75 mV PSP.

## The input

Each cell receives `N_E=250` excitatory (and, in the E+I condition, `N_I=220` inhibitory)
presynaptic Poisson spike trains. Two independent parameters control how correlated the two
cells' *inputs* are with each other:

- **`p`** — literal sharing: a fraction `p` of each cell's `N` inputs are the *exact same*
  spike train, delivered identically to both cells.
- **`r_in`** — statistical correlation: the remaining `(1-p)N` inputs are drawn from a single
  shared "mother-train" pool (thin one high-rate parent Poisson process per child, then
  jitter each spike — the Kuhn/Aertsen/Rotter method). Every pair among these pooled
  children is correlated at `r_in` — including two inputs onto the *same* cell, not just
  matched cross-cell pairs. This detail matters: it's what makes the amplification in
  M-Eq(1) come out right (see `PROGRESS.md`'s Phase 3 derivation for why the naive
  "index-paired" reading would silently fail to reproduce it).

## The output — what's measured

- **`c`** — Pearson correlation of the two cells' total synaptic current traces
  (`i_syn_a_mV`, `i_syn_b_mV`).
- **`r_out`** — Pearson correlation of the two cells' *output spike* trains, via a `T=50ms`
  sliding-window rate estimate (S-Eq 34–37).

Both come from the *same shared measurement pipeline* (`src/analysis.py`, one level up) that
every block and every level (L1–L4) of this project uses identically — deliberately not
block1-scoped, so a later block/level's numbers stay comparable to this one's.

## Objectives — what we're looking for

This block builds the core intuition the rest of the paper depends on: does shared/correlated
*input* drive output *correlation*, and can matched inhibition cancel that effect?

- **Fig. 1B** (`p` swept `0→0.4`, E-only, `r_in=0`): `c` and `r_out` should grow roughly
  linearly with `p`, staying moderate (`≲0.4` at `p=0.4`).
- **Fig. 1E** (`r_in` swept, `p=0.2` fixed): the **E-only** curve should rise steeply
  (`r_out → ~1` by `r_in≈0.1`) — weak input correlations get massively amplified by summing
  over `N=250` inputs (M-Eq 1: `c ≈ p + N·r_in`). The **E+I** curve should stay strongly
  suppressed across the same range — matched inhibitory correlation cancels the
  amplification. **This E-only-vs-E+I contrast is the paper's central mechanism**, reproduced
  here at the smallest possible scale (2 neurons, purely feedforward) before Blocks 2/3 show
  it survives in full recurrent networks.
- **Fig. 1C / 1F**: example single-trial traces at `p=0.2, r_in=0.025` — visually showing
  simultaneous E/I current excursions that cancel in the total current (1F specifically).

## Module map

One main script (`full_pass.py`) plus focused helpers, all living in this directory —
no separate top-level `scripts/`.

| File | Role |
|---|---|
| `calibration.py` | Closed-form + Brian2-verified synaptic weight calibration (`J_E`, `J_I`) |
| `model.py` | The two-neuron pair model (`simulate_pair`) — the tested foundation every other module builds on; also used directly for small one-off runs (calibration, illustrative traces) |
| `current_trace.py` | Precomputed synaptic current traces (ADR 0002), replacing per-event Brian2 objects — needed at the paper's real input volume |
| `dataset.py` | Input generation — literal sharing (`p`) + mother-train correlation (`r_in`) |
| `calibrate_rate.py` | Numerically calibrates the E-only input rate (an ambiguity the paper leaves unstated) |
| `batched_model.py` | `simulate_pairs_batch` — N independent pairs (e.g. all 15 sweep points) in ONE Brian2 `NeuronGroup(2N)` instead of N separate runs; exact, not an approximation, since the points don't couple (see module docstring). This is what makes the production sweep GPU-friendly. |
| `full_pass.py` | **Main script.** Builds the Fig. 1B/1E sweep grid, runs it via `batched_model` (chunked over time), writes a structured CSV. `python -m block1.full_pass` |
| `eval.py` | Qualitative trend checks against the paper's stated results |
| `plot_fig1.py` | All four Fig. 1 plots: 1B/1E from `full_pass`'s CSV (combined panels, `c` dashed / `r_out` `-o-`), 1C/1F bespoke illustrative traces (raster / current / V, scale bars) |

`eval.py`'s trend checks and small-scale correctness sweeps that predate the batched
production path are exercised through `model.simulate_pair` directly in the test suite
(`tests/block1/test_model.py`, `test_model_chunking.py`, `test_integration.py`) rather
than through a separate sweep-runner module.

## Status

See the project root's `PROGRESS.md` for the authoritative, up-to-date status — what's been
run, at what scale, with what actual numbers. `docs/adr/0001` and `0002` record the two
hard-to-reverse decisions behind this block (Brian2 as the simulation framework; precomputed
current traces over per-event objects for scalability).
