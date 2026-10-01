# Block 1 — Feedforward Postsynaptic Pair (Fig. 1)

Reproduces main-text Fig. 1 of Renart et al. 2010, *The Asynchronous State in Cortical
Circuits* (*Science* 327:587): two independent current-based LIF neurons driven by
partly-shared, partly-correlated Poisson input. This is the smallest scale at which the
paper's central mechanism appears — weak input correlation is hugely amplified by summing
over many inputs, and matched inhibition cancels that amplification.

**Status: complete.** All four panels have been produced. See the root `PROGRESS.md` for
the actual numbers from the last run.

## Run it

Everything runs from the **repo root** with `PYTHONPATH=src`. Nothing here needs the
cluster; the full pass is long but single-process.

```bash
# 1. The sweep (this is the expensive one -- see Cost below)
PYTHONPATH=src .env/bin/python -m block1.run sweep

# 2. Panels 1B and 1E, from the CSV the sweep just wrote
PYTHONPATH=src .env/bin/python -m block1.plot

# 3. Panels 1C and 1F -- illustrative traces, fresh short simulation, ~seconds
PYTHONPATH=src .env/bin/python -m block1.plot        # 500 ms window
PYTHONPATH=src .env/bin/python -m block1.plot 1000   # or pick your own
```

Outputs land in `artifacts/`: `block1_full_pass.csv`, then `fig1b.png`, `fig1e.png`,
`fig1c.png`, `fig1f.png`.

## What each panel is, and what "correct" looks like

| Panel | Command | Paper's result |
|---|---|---|
| **1B** | `plot_fig1` | `c` and `r_out` both grow roughly linearly with the shared fraction `p`, staying below about 0.4 at `p = 0.4` |
| **1E** | `plot_fig1` | The E-only curve rises steeply, `r_out` approaching 1 by `r_in ≈ 0.1`. The E+I curve stays strongly suppressed over the same range. **This contrast is the whole point of the block.** |
| **1C** | `plot_fig1_traces` | E-only example trial at `p = 0.2`, `r_in = 0.025` |
| **1F** | `plot_fig1_traces` | Same point with I added. E and I currents excurse together and cancel in the total |

## Changing the settings

Every number lives in `config.yaml` under `pair_model`. Nothing is hardcoded in Python.

**To trade accuracy for time** — this is the knob you want first:

| Key | Paper value | Effect |
|---|---|---|
| `simulation.length_s` | `10000.0` | Simulated seconds per sweep point. The dominant cost, and what sets the noise floor on every `c` and `r_out`. Halving it roughly halves runtime and widens the error by about √2. |
| `simulation.chunk_duration_s` | `500.0` | Memory/runtime trade only, **not** accuracy. Chunking is mathematically exact (`analysis.StreamingCorrelation`). Lower it if you hit memory pressure. |
| `simulation.dt_ms` | `0.05` | Integration step. Do not raise it without re-running the calibration below. |

**To change what is swept:**

| Key | Paper value | Meaning |
|---|---|---|
| `sweeps.shared_fraction_grid` | `[0.0, 0.1, 0.2, 0.3, 0.4]` | Fig. 1B's `p` points |
| `sweeps.r_in_grid` | `[0.0, 0.01, 0.025, 0.05, 0.1]` | Fig. 1E's `r_in` points |
| `sweeps.p_fixed` | `0.2` | `p` held fixed for 1C, 1E, 1F |
| `sweeps.r_in_fig1f_example` | `0.025` | The example point 1C and 1F illustrate |

Adding grid points costs time linearly but they all batch into one Brian2 run per chunk,
so the marginal cost of one more point is small compared with one more second of `length_s`.

**Model parameters** (`pair_model.neuron`, `.synapse`, `.inputs`) are all transcribed from
SOM S-p.19. Changing any of them means you are no longer reproducing the paper.

## The one calibrated value

`inputs.rate_e_only_calibrated_hz` (`4.04297` Hz) is **not** from the paper. The paper
states 20 Hz for the E+I condition but never states the E-only input rate. It is derived
by bisection so that the E-only cell fires at 5 Hz output when `r_in = 0`, which is the
condition the paper's Fig. 1E curve implies. `calibrate_rate.py` holds that derivation.

If you change `n_excitatory`, `tau_m_ms`, `epsp_peak_mV`, or the threshold, **this value is
stale** and must be re-derived before the E-only curve means anything.

`J_E` and `J_I` are also calibrated rather than read from config: `calibration.py` solves
numerically for the weights that give a ±0.75 mV single-spike PSP, which is what the paper
specifies. That happens automatically on every run.

## Cost

At the paper's `length_s = 10000.0`, all 21 sweep points batch into one Brian2 `NeuronGroup`
of 42 neurons (two cells per point) and run chunk by chunk. The dominant cost is Brian2's
per-timestep work over the full 10,000 simulated seconds at `dt = 0.05 ms`. Consult `PROGRESS.md` for the last
measured wall-clock time on this machine rather than trusting an estimate here.

For a quick check that the pipeline works, set `length_s` to something like `100.0`. The
curves will be visibly noisy but the E-only-versus-E+I contrast should already be obvious.

## Module map

| File | Role |
|---|---|
| `model.py` | The two-neuron pair — the paper's equations, and the tested foundation everything else builds on |
| `inputs.py` | Input generation: literal sharing (`p`) plus mother-train correlation (`r_in`) |
| `calibration.py` | The two values the paper specifies indirectly or not at all: the synaptic weights, and each condition's input rate |
| `run.py` | **The entry point.** `sweep`, `calibrate`, `check` |
| `plot.py` | All four panels |

Machinery lives in `src/lib`, not here: the synaptic current filter (`psc`), the Brian2
batching and time chunking (`brian_batch`, `chunking`), the trend check (`checks`) and the
shared plotting helpers (`plotting`).

## ⚠ Before running the full pass: one value must be derived first

`block1.run` **raises** until you derive the E+I input rate:

```bash
PYTHONPATH=src .env/bin/python -m block1.run calibrate e_plus_i
```

then paste the printed value into `config.yaml` as `rate_e_plus_i_calibrated_hz`.

**Why.** The supplement says "The input firing rate was set to 20 spikes/s **to produce an
output rate of 5 spikes/s** when r_in = 0". Measured with this repo's parameters, 20 spikes/s
produces **12.05 ± 2.57** spikes/s. The PSP calibration, the analytic mean drive and the
free-membrane statistics each check out independently, so this is not a simulation artefact —
the paper's two numbers are not consistent with each other here. The decision (2026-09-30) is
to honour the stated *output* rate and recalibrate the input, exactly as this project already
resolved the unstated E-only rate, so that both Fig. 1E curves share an operating point, which
is what the caption means by "identical statistics". The code raises rather than silently
falling back to 20 spikes/s, which would leave the two curves at ~5 and ~12 spikes/s.

## Fixed 2026-09-30: the shared inputs were not mother-train children

`dataset.py` drew the `p·N` literally-shared trains as **independent** Poisson processes, so
they carried none of the mother train's correlation. The supplement says "**Each**
pre-synaptic train was a thinned version of the mother train". This biased the current
correlation low by about 10% at `r_in = 0.01` and 5.5% at `r_in = 0.025`, the marked circle
of Figs. 1C and 1F. All trains now come from one mother pool, pinned by a regression test that
fails on the old construction.

**Any Fig. 1E produced before 2026-09-30 carries that bias** and should be regenerated once
the E+I rate above is derived. Fig. 1B (`r_in = 0`) is unaffected.

## Still open

**The `r_in` grid covers a quarter of the paper's axis.** Fig. 1E sweeps `r_in` over
`(0, 0.4)`; `sweeps.r_in_grid` stops at `0.1`. Widen it to reproduce the full panel.

## Known deviations from the paper

1. **The E-only input rate is calibrated, not stated** (above).
2. **Current is in mV, not nA.** The model is current-based with `V` relative to rest, so
   the natural unit here is mV. The paper's 1C/1F scale bar is in nA. Shape and correlation
   are unaffected; only the axis label differs.
3. **The 1C/1F raster shows a subsample of individual trains.** The model pools all ~250
   to 470 arrivals into one stream internally, which would render as a solid block. The
   displayed trains are drawn at the same rate and `r_in`, so the correlation structure is
   representative, but they are not literally the trains driving the plotted traces.
