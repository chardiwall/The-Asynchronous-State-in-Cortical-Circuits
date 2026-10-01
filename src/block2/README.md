# Block 2 — Recurrent Binary Network (Fig. 2)

Reproduces main-text Fig. 2: the analytically tractable binary network where `r̄ ~ 1/N`,
E/I tracking and current-correlation cancellation are *derived* rather than just observed.
Block 1 showed the mechanism with two neurons and imposed input correlations; block 2 shows
it emerging spontaneously from a recurrent network's own dynamics.

**Panels that need simulation: 2B, 2C, 2D, 2E, 2G.** Panels 2A and 2F are schematics in the
paper (2F is captioned "Description of the asynchronous self-consistent solution"), so
nothing is generated for them.

## Run it

All commands are from the **repo root** with `PYTHONPATH=src`.

### Step 1 — validate against the closed-form theory (local, minutes)

```bash
PYTHONPATH=src .env/bin/python -m block2.run explore
```

Small `N`, short runs, checked against S-Eq(18)'s predicted rates and S-Eq(28–29)'s
correlations. Writes `artifacts/block2_exploratory_pass.csv`. Run this after any change to
the model, before spending cluster time. Expect `X`'s rate to land on `m_X = 0.1` almost
exactly, and `E`/`I` to sit near the prediction but off by a finite-size amount that
shrinks as `1/√N` — that deviation is expected, not a bug.

### Step 2 — Fig. 2C's two sweeps (cluster, long)

Fig. 2C needs both: firing correlations from one job, current correlations from another.
Both walk the same `(N, realisation)` grid, 347 tasks at the config's defaults.

```bash
# Firing correlations (r-bar). Submit PER SIZE TIER -- see the memory note below.
sbatch --array=0-299%20  --mem=1G slurm/block2_sweep.slurm
sbatch --array=300-324%20 --mem=3G slurm/block2_sweep.slurm
sbatch --array=325-336%19 --mem=6G slurm/block2_sweep.slurm
sbatch --array=337-346%13 --mem=9G slurm/block2_sweep.slurm

# Current correlations (c, c_EE, c_II, c_EI). Memory is flat across N here.
sbatch --mem=6G slurm/block2_sweep_current.slurm

# Once BOTH jobs have fully finished:
PYTHONPATH=src .env/bin/python -m block2.run aggregate
PYTHONPATH=src .env/bin/python -m block2.run aggregate
```

A single task can be run directly, which is how to test before submitting:
`PYTHONPATH=src .env/bin/python -m block2.run sweep 0`

### Step 3 — the illustrative panels (local, minutes)

```bash
for panel in b d e g; do
  PYTHONPATH=src .env/bin/python -m block2.run panels $panel
done
```

### Step 4 — plot everything

```bash
PYTHONPATH=src .env/bin/python -m block2.run plot
```

Writes `fig2b/c/d/e/g.png` to `artifacts/block2_illustrative/`. 2C reads the two aggregated
CSVs from step 2; the rest read step 3's per-panel CSVs.

## What "correct" looks like

| Panel | Paper's result |
|---|---|
| **2B** | `Total` current hovers near the dashed threshold while its E, X and I components individually sit far from it — the dynamic balance |
| **2C** | `r̄` falls along `1/N`, `c` along `1/√N`, while `c_EE`, `c_II`, `c_EI` stay `O(1)`. **The central result of the block** |
| **2D** | I visibly tracks E with a small lag that shrinks as `N` grows |
| **2E** | Component CCGs peak near zero lag; the EI/IE peak offset shrinks with `N` |
| **2G** | Histogram of `r` over EE pairs is wide and centred near zero: `σ_r ≫ r̄` |

## Changing the settings

Everything is under `binary_network` in `config.yaml`.

**Cost versus accuracy** — the knobs that matter:

| Key | Paper value | Effect |
|---|---|---|
| `full_pass.length_tau` | `200000` | Simulated neuronal time constants per realisation. Sets the noise floor on every correlation. The single biggest cost lever. |
| `full_pass.sizes` | `[100 … 8192]` | Fig. 2C's `N` grid. Cost is close to `O(N²)` per realisation, so the largest one or two sizes dominate the entire job. |
| `full_pass.repeats` | `[50 … 10]` | Realisations averaged per dot. The paper uses 50 everywhere; this tapers at the top of the range where cost is worst. Raise the tail back to 50 for a closer match, at large expense. |
| `burn_in_tau` | `10` | Discarded before recording. Not from the paper — the paper states no burn-in. |

**Model parameters** (`couplings`, `connection_probability`, `m_x`, `theta`, `tau_ms`) are
transcribed from SOM S-p.18–19. Changing them means you are no longer reproducing Fig. 2.

**Illustrative panel windows** (`illustrative_panels.*`) control only how long each
qualitative panel runs. They are deliberately far shorter than the full pass because those
panels show shape, not ensemble precision. Raise them if a panel looks too noisy to read.

`sizes_fixed` (`[1024, 8192]`) drives 2D directly and supplies the largest size for 2B, 2E
and 2G. Change it and all four follow.

## Memory: read this before submitting

This block has cost several days to memory problems. Two things are now true and should
stay true:

1. **`--mem` must be set per size tier** for `block2_sweep.slurm`. A flat value sized for
   `N=8192` throttles the whole grid to about 11-way concurrency when most tasks need
   megabytes; a value too small causes real OOM kills, not queuing. `block2_sweep_current.slurm`
   is different — it records a fixed-size subsample, so one `--mem=6G` covers every `N`.
2. **Size `--mem` from a ground-truth measurement** (`/usr/bin/time -v` over a *complete*
   run), never from a short snapshot. Short snapshots underestimated the true peak twice in
   this project's history and both times produced OOM kills after many hours of compute.

`connectivity.build_weights_stacked` exists for this reason: building six `(n,n)` matrices
and hstacking them left roughly twice the final weight size resident for the life of the
process even after `del`, because glibc did not return it to the OS.

## Module map

| File | Role |
|---|---|
| `model.py` | The paper's binary network in one place: S-Eq(5-7)'s Glauber dynamics, S-Eq(18)'s closed-form rate prediction, and the pure-Python reference simulation they are checked against. **The correctness oracle** the JIT path is validated against |
| `connectivity.py` | S-Eq(1) connectivity in two memory layouts, plus the couplings helper |
| `measure.py` | The correlation, CCG and current-component decomposition every panel reuses |
| `panels.py` | Raw data for the illustrative panels 2B/2D/2E/2G |
| `run.py` | **The entry point.** `explore`, `sweep`, `current`, `aggregate`, `panels`, `plot` |
| `plot.py` | Panels 2B, 2D, 2E, 2G |
| `plot_sweep.py` | Panel 2C — different in kind, it reads the two aggregated sweep CSVs |

Machinery lives in `src/lib`: the Numba-JIT'd production path (`glauber`, `glauber_panels`),
the Slurm-array bookkeeping (`tasks`) and the shared plotting helpers (`plotting`).
Cluster scripts live in `slurm/`.

## Fig. 2C's decomposition, corrected 2026-09-30

This block previously computed the wrong quantity for `c_EE`, `c_II` and `c_EI`, and any
`artifacts/block2_full_pass_current.csv` produced before 2026-09-30 is **invalid**. Delete it
and rerun `block2_sweep_current.slurm`.

The main text (p.588): "Because the synaptic current to each cell consists of an excitatory
and an inhibitory **component**, the average current correlation across cell pairs, `c`, can
be decomposed into `c_EE`, `c_II`, and `c_EI`", cross-referenced to Fig. 1F's red and green
traces, which are one cell's E and I current components. So `c_EE` is the correlation between
the **E-components** of two cells' currents. The old code instead correlated *total* currents
between pairs drawn from different postsynaptic populations and summed those as though the
identity applied, giving a total 4× too large with every component positive.

Now: each recorded cell's E, I and X components are correlated component-wise under a
**common total-current normalisation**, which is what makes the decomposition an identity
rather than six unrelated coefficients. Six components are reported, `c_EE`, `c_II`, `c_XX`,
`c_EI`, `c_EX`, `c_IX`, and

```
c = c_EE + c_II + c_XX + 2·c_EI + 2·c_EX + 2·c_IX
```

holds exactly. A test asserts that identity, and it was checked on real simulated output:
at `N = 120` and `N = 480` the components come out positive and growing with `N` while `c_EI`
is large and negative and the total decays, which is all three of the paper's qualitative
claims.

**The lagged form needs all nine cross-terms, not six.** `c_IE(lag) = c_EI(-lag)`, so the
two coincide only at zero lag; away from it the total needs `c_EI + c_IE`, not `2·c_EI`.
The paper makes the same distinction, magnifying "the IE and EI CCGs" as two separate curves
in Fig. 2E's inset — that asymmetry about zero *is* the EI-Lag. Measured on a real run at
lag +1 sample: `c_EI = −0.0787` against `c_IE = −0.0290`. A test asserts the nine sum to the
total at every lag.

Two further choices worth knowing: the average runs over pairs of **E cells** (the population Fig. 2G
also uses; the paper says only "across cell pairs"), and `c_XX` is reported because Fig. 2C's
own legend shows it, which is itself the clue that the decomposition is over components, since
a `c_XX` cannot exist under the postsynaptic-population reading.

## Known deviations from the paper

0. **Fig. 2C plots `|c_EI|`, which the published panel does not plot at all.** Read off the
   paper's own Fig. 2C: it shows `c_EE`, `c_II`, `c_XX` as flat coloured squares, plus `c`
   and `r̄`. `c_EI` is large and *negative*, so a log axis cannot show it and the paper
   discusses it only in the text. This repo plots it as a magnitude with a hollow marker,
   which is more informative but is a deviation from the panel as printed. The CSV keeps
   the true sign.
1. **Burn-in and initial condition are not from the paper.** Each neuron starts as an
   independent Bernoulli(0.5) draw and `burn_in_tau × 3N` ticks are discarded. The paper
   states neither.
2. **`c_EI` is plotted as `|c_EI|` in Fig. 2C.** It is large and *negative* — that is what
   cancels `c_EE + c_II` — and a log axis cannot show a negative value. The magnitude is
   plotted and the label says so. The stored CSV keeps the true sign.
3. **Fig. 2E's subsample size reuses `fig2g_subsample_neurons`.** The paper states 1000 E
   cells for 2G but states no subsample size for 2E's CCGs. Reusing it is a choice, not a
   transcription.
4. **The repeat count tapers at large `N`** (50 down to 10) against the paper's flat 50,
   to fit the researcher's wall-clock budget. `full_pass.repeats` reverses this.
