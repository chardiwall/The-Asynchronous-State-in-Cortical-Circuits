# Block 3 — Recurrent Conductance-Based Spiking Network (Fig. 3)

Reproduces main-text Fig. 3: the conductance-based integrate-and-fire version of block 2's
binary network, with the same architecture but realistic population sizes
(`N_E = 4000`, `N_I = 1000`, `N_X = 4000`). This is the repo's main reproduction target.
Fig. 3C–D is the paper's **experimentally testable prediction** and the reason the block
matters: it shows the cancellation directly in the membrane potential.

Equations are declared literally against SOM S-p.20–21 as a Brian2 model: one neuron
equation shared by the E and I groups, one difference-of-exponentials synapse equation
shared by all six ordered population pairs, and `X` as a plain Poisson group. This file is
the run guide.

## ⚠ Things to check before trusting Fig. 3

These are recorded here because they are inferences, not transcriptions.

1. **The `I_app` sweep is incomplete in the source.** Both the main text and the SOM print
   it with ellipses: `Iapp = −1.3, −0.65, . . . , −0.1, 0, 0.2, 0.74, 1.48, . . . 3.7 nA`.
   The omitted levels are stated nowhere, and the printed values fit neither an arithmetic
   nor a geometric series, so they cannot be reconstructed. `config.yaml` carries **only the
   eight printed values**. Fig. 3D will therefore have eight points where the paper's has
   more. Add levels to `fig3cd.i_app_nA` if you can source them; do not let me guess them.

2. **Ten recorded cells per condition per network is derived, not stated.** The paper gives
   only two numbers: 450 pairs for each same-condition curve and 1000 for the EPSP–IPSP
   curve. Holding 10 cells at each level per network reproduces both exactly —
   `C(10,2) = 45` same-condition pairs × 10 networks = 450, and `10 × 10 = 100` cross pairs
   × 10 networks = 1000. No other simple scheme gives both. If you
   reject it, only `fig3cd.n_recorded_cells_per_condition` and the pair enumeration change.

3. **Fig. 3B runs at 1000 s, not the 5000 s the supplement states.** A deliberate,
   documented deviation (decided 2026-10-01, GitHub issue #8). The panel's claim is that the
   measured histogram is wide *relative to its jittered null*, and the null's width is pure
   estimator noise. Measured, for independent 1 Hz trains at the stated count window:

   | run length | null width | rate matrix | wall clock |
   |---|---|---|---|
   | 200 s | 0.0114 | 1.6 GB | 27 h |
   | **1000 s (configured)** | **0.0056** | **8 GB** | **135 h** |
   | 5000 s (the supplement's) | 0.0027 | 40 GB | ~28 days |

   On an axis spanning ±0.05, the 200 s null is as wide as the measurement and the panel
   says nothing. At 1000 s the comparison is real, but the separation is visibly weaker than
   the published figure's. The population-averaged correlation itself averages over ~500k
   pairs and is unaffected at any of these lengths; only the spread is.

   Raising `panels.length_s` back to 5000 makes `panels.py` refuse before building the
   network, by design, rather than fail after days with nothing written.

A fourth, smaller one: **which `I_app` level is the EPSP curve and which the IPSP curve is
inferred** from the extremes of the swept list. The SOM states the intent ("adjusted to
isolate the EPSPs and IPSPs in their respective reversal potentials") but never maps a
level to a curve. `fig3cd.epsp_condition_nA` and `ipsp_condition_nA` hold that reading.

## Run it

From the **repo root** with `PYTHONPATH=src`. This network has about 9 million synapses, so
only the smoke test is local.

### Step 0 — smoke test (local, about 25 minutes)

```bash
PYTHONPATH=src .env/bin/python -m block3.run networks 0 --seconds 5
```

Full paper-scale `N` over a short window. Proves the model and the shared analysis pipeline
are wired correctly: no NaN, non-zero and non-runaway rates. **Not** a statistically
meaningful `r̄`. Note the `N` is *not* shrunk for this: block 3's conductances are fixed nS
values rather than `1/√N`-scaled, so a smaller network would starve every neuron of input
and land in a different dynamical regime, not a cheaper version of this one. Shortening the
duration is the only cheap knob.

### Step 1 — Fig. 3A–B statistics (cluster, days)

```bash
sbatch slurm/block3_networks.slurm                                  # 10 networks
PYTHONPATH=src .env/bin/python -m block3.run aggregate     # when all 10 land
```

Writes `artifacts/block3_full_pass.csv`: per-network E and I rates and `r̄`.

### Step 2 — Fig. 3A and 3B panel data (cluster, one long run)

```bash
sbatch slurm/block3_panels.slurm
```

One network, one run, both panels, at `panels.length_s` (5000 s — see item 3 above; this is
the single most expensive job in the repo). Writes `fig3a_raster.csv`, `fig3a_tracking.csv`
and `fig3b_correlations.csv` to `artifacts/block3_panels/`. Locally with a short window:
`PYTHONPATH=src .env/bin/python -m block3.run panels --seconds 5`

### Step 3 — Fig. 3C–D (cluster, 90 tasks)

```bash
sbatch slurm/block3_vm_ccg.slurm
PYTHONPATH=src .env/bin/python -m block3.run aggregate
```

90 tasks = 10 networks × (8 current levels + 1 cross condition), each 50 s of simulated
time. A single condition can be run directly, which is how to test before submitting:

```bash
PYTHONPATH=src .env/bin/python -m block3.run vm 0 -1.3 3.7   # network 0, EPSP vs IPSP
PYTHONPATH=src .env/bin/python -m block3.run vm 0            # or by flat task index
```

### Step 4 — plot

```bash
PYTHONPATH=src .env/bin/python -m block3.run plot
```

## What "correct" looks like

| Panel | Paper's result |
|---|---|
| **3A** | E rate **1 spike/s**, I rate **3.6 spikes/s**. Raster irregular, no population bursts. The three z-scored activity curves visibly track each other |
| **3B** | Histogram wide and centred near zero with `r̄ < 0.001`, and **barely distinguishable from its jittered surrogate** — that near-identity is the asynchronous-state claim |
| **3C** | Large positive CCG for EPSP–EPSP and for IPSP–IPSP, large **negative** for EPSP–IPSP, near zero at rest |
| **3D** | A **V-shape**: peak correlation positive at both reversal potentials, minimum near rest |

The last measured full-pass network gave E = 1.050 Hz and `r̄ = 0.000125`, both on target.
I came out somewhat above 3.6 Hz on that single network; the paper's figure is a
population average, so check it across more networks before treating it as a discrepancy.

## Changing the settings

Everything is under `spiking_network` in `config.yaml`.

**Cost versus accuracy:**

| Key | Paper value | Effect |
|---|---|---|
| `simulation.length_s` | `200.0` | Simulated seconds per network, for Fig. 3A–B. Dominant cost and the noise floor on `r̄`. Fig. S6 uses 5000 s |
| `simulation.n_networks` | `10` | Networks averaged in Fig. 3A–B |
| `fig3cd.length_s` | `50.0` | Per Fig. 3C–D condition. Stated by the SOM |
| `fig3cd.n_networks` | `10` | Stated by the SOM |
| `r_bar_sample_size` | `1000` | E neurons subsampled for `r̄` and for Fig. 3B's histogram. Mirrors Fig. S6's own "1000 E and 1000 I cells". Cost here is `O(n²)` in pairs |
| `burn_in_ms` | `100.0` | **Not from the paper** — it states no warm-up period |
| `simulation.dt_ms` | `0.05` | Stated. Delays are quantised to this grid |

**Model parameters** (`neuron`, `conductances_nS`, `synapse`, `delays_ms`,
`external_input`, `connection_probability`, `populations`) all come from SOM S-p.20–21.
Changing them means you are no longer reproducing Fig. 3.

**Panel settings:** `fig3a.raster_neurons` (500), `fig3a.tracking_bin_ms` (3.0),
`fig3b.jitter_ms` (500.0) are all stated in the Fig. 3 caption. `fig3b.n_surrogate_sets`
is 1 because the grey histogram is one surrogate of the same pairs, not the 500-surrogate
significance test of S-Eq(41), which is a different procedure for *in vivo* data.

## Compute reality

Brian2's runtime mode needs its Cython headers, which this cluster lacks and there is no
`sudo`. Both Slurm scripts therefore run inside a pyxis/enroot container that installs
`python3.12-dev` per task. Without it Brian2 silently falls back to pure-Python codegen,
which measured **29× slower** and makes the full pass infeasible.

Measured reference on this node: 200 s of simulated time cost **27h07m** wall and **3.85GB**
peak RSS. A 200 ms benchmark had extrapolated 67 h, overestimating by about 2.5×, because
per-timestep fixed overhead amortises far better over a long run. Size new jobs from the
27-hour figure, and confirm with the `/usr/bin/time -v` log each task writes.

One cost in that figure is **unmeasured**: `measure.pairwise_correlations` builds a
`(1000, ~49,950)` float64 rate matrix (~400 MB) and hands it to `numpy.corrcoef`, which
allocates a centred copy plus the Gram matrix. The 3.85 GB run predates that step, so
measure peak RSS once before sizing `--mem` rather than repeating this project's history of
OOM kills from unvalidated estimates.

`cpp_standalone` was benchmarked and gave **no speedup** (possibly a regression, not root
caused). Do not reach for it without re-measuring.

## Module map

| File | Role |
|---|---|
| `model.py` | The Brian2 network: E/I `NeuronGroup`s, X `PoissonGroup`, six `Synapses`. `theta` is per-neuron so Fig. 3C-D can disable spiking in the recorded cells |
| `connectivity.py` | Per-synapse conductance heterogeneity and delays, plus the Bernoulli draw |
| `measure.py` | This block's analysis: rates, spike times, subsampling, the pairwise-correlation vector and its mean, plus Fig. 3C-D's pair enumeration and CCG peak |
| `figures_ab.py` | What one network run records for Fig. 3A and 3B |
| `figures_cd.py` | Fig. 3C-D: spiking disabled, `I_app` injected, membrane-potential CCGs |
| `run.py` | **The entry point.** `networks`, `panels`, `vm`, `aggregate`, `plot` |
| `plot.py` | All four panels |

Machinery lives in `src/lib` (`tasks`, `plotting`); cluster scripts live in `slurm/`.

## Known deviations from the paper

1. **The three items at the top of this file** — the incomplete `I_app` list, the derived
   recorded-cell count, and the EPSP/IPSP level assignment.
2. **No warm-up period is stated**; `burn_in_ms = 100.0` (about 7 membrane time constants)
   is this project's choice.
3. **X's conduction delay range is assumed** to follow the excitatory range, since
   `V_rev^X = V_rev^E` makes X excitatory-type. The paper gives ranges only "from excitatory
   cells" and "from inhibitory cells".
4. **Fig. 3A's "500 E and I neurons" is read as 500 total**, split between E and I in
   proportion to population size. The caption does not say whether it means 500 of each.
5. **The synaptic drive is first-order in time, despite `rk2`.** The paper specifies
   second-order Runge-Kutta and every group here requests it, but Brian2's `(summed)`
   mechanism computes `I_E`/`I_I`/`I_X` from the start-of-step `V` and holds them constant
   across the step. So `rk2` applies to the leak plus those frozen currents while the
   conductance term is effectively Euler. This is inherent to `(summed)`, not a choice, and
   at `dt = 0.05 ms` against `τ_r = 1 ms` it is the dominant discretisation error.
6. **The membrane-potential CCG is a Pearson correlation coefficient**, sampled at 1 ms.
   The SOM says only "we computed cross-correlograms of the voltages" and gives no formula
   or bin; S-Eq(42)'s spike-train normalisation by `ν_i ν_j` has no meaning for a voltage.
   A correlation coefficient is the reading consistent with Fig. 3D plotting a peak height.
7. **Cells with spiking disabled stop driving their targets.** With 10 cells per condition
   out of 4000 the perturbation is small but not zero. Each task runs one condition pair
   rather than all levels at once, to keep it that way.

## Out of scope

Supplementary Figs. S6, S7 and S8 (detailed characterisation, robustness sweeps, sinusoidal
drive). None of them needs a change to `model.py` — only a different protocol and analysis.
