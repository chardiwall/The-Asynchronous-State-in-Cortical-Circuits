# The Asynchronous State in Cortical Circuits — reproduction

A standalone reproduction of the three main-text figures of

> A. Renart, J. de la Rocha, P. Bartho, L. Hollender, N. Parga, A. Reyes, K. D. Harris,
> *The Asynchronous State in Cortical Circuits*, **Science 327**, 587 (2010),
> and its Supporting Online Material.

The paper's claim is that a recurrent network of excitatory and inhibitory neurons can fire
at realistic rates while remaining almost uncorrelated, because correlated excitatory and
inhibitory input currents cancel. This repository implements that argument as three
simulations and measures the same quantities the paper measures.

## What is measured

One invariant runs through all three blocks:

- `r̄` — the population-averaged pairwise spiking correlation,
- `σ_r` — the width of its distribution,
- `c_EE`, `c_II`, `c_EI`, `c` — the current-component correlations whose cancellation is
  the mechanism,
- and how each scales with network size `N`.

The same estimators are used everywhere (`src/analysis.py`), so a result in one block is
comparable with a result in another.

## The three blocks

| Block | Figure | Model |
|---|---|---|
| [`src/block1/`](src/block1/README.md) | Fig. 1 B/C/E/F | Feedforward pair of current-based LIF neurons sharing a fraction of their inputs — separates the two sources of output correlation and shows E–I input correlation *decorrelates* |
| [`src/block2/`](src/block2/README.md) | Fig. 2 B/C/D/E/G | Recurrent binary network under Glauber dynamics — the analytically tractable model, where the `r̄ ~ 1/N` scaling and the cancellation are *derived* rather than only observed |
| [`src/block3/`](src/block3/README.md) | Fig. 3 A/B/C/D | Recurrent conductance-based spiking network at realistic sizes (`N_E = 4000`, `N_I = 1000`, `N_X = 4000`) — the main reproduction target; Fig. 3C–D is the paper's experimentally testable prediction |

Each block holds the paper's own equations and exactly **one entry point**, `run.py`, with
subcommands. Anything that is machinery rather than science lives in `src/lib`, and the
shared estimators live in `src/analysis.py`. Dependencies run in whichever direction avoids
restating the paper: `lib/glauber` imports block 2's S-Eq(1) connectivity rather than
duplicating it, and a block may import another block where that is genuinely the same
quantity.

Fig. 2A and 2F are schematics in the paper, not simulations, so nothing is generated for
them. Supplementary figures are out of scope.

## Status

All three blocks are implemented and under test. Results are a separate matter: the large
passes are multi-day cluster jobs, and `artifacts/` is not tracked, so **a fresh clone ships
the code, not the figures.**

| Block | Implementation | Results produced here |
|---|---|---|
| 1 | complete | Fig. 1 B/C/E/F, at the paper's full `length_s = 10000` |
| 2 | complete | Panels 2B/2D/2E/2G. Fig. 2C's two `N`-sweeps are long Slurm-array jobs, not yet run to completion |
| 3 | complete | Exploratory pass only; the Fig. 3A–B and 3C–D full passes have not been run |

Each block's own README is the run guide: exact commands per panel, which `config.yaml` keys
trade cost against accuracy, the paper's target result, and that block's known deviations
from the paper. Read those before trusting any number.

Two values in Fig. 3C–D's configuration are inferences rather than transcriptions, and are
flagged at the top of [`src/block3/README.md`](src/block3/README.md).

## Layout

| Path | Contents |
|------|----------|
| `config.yaml` | Every seed, model parameter and output path, with per-block provenance back to the paper's SOM. Nothing numeric and no path is hardcoded in Python |
| `src/analysis.py` | The shared measurement pipeline, used identically by every block |
| `src/config.py` | The `config.yaml` loader and `output_path`, the one place output locations are resolved |
| `src/lib/` | Machinery: Numba kernels, Brian2 batching, chunking, Slurm-array bookkeeping, plotting helpers |
| `src/block1/`, `src/block2/`, `src/block3/` | One directory per figure, each with its own README |
| `slurm/` | Cluster job scripts, named by block and figure |
| `tests/` | Tests, written before the code they cover |
| `artifacts/` | Metrics, logs and figures (generated; not tracked) |

## Running it

```bash
pip install -r requirements.txt
pytest                                  # 169 tests
python -m block1.run sweep              # see each block's README for the rest
```

`pytest.ini` sets `pythonpath = src`, so the modules import as `analysis`, `config`,
`block1`, `lib`, and so on. There is no packaging step — this is a reference
implementation to read, run and cite against the paper, not a library to install.

## Conventions

Worth knowing before changing anything:

- **Every numeric constant comes from `config.yaml`**, with a comment tracing it to the
  paper or marking it explicitly as an inference or a guard. If a value is not in
  `config.yaml`, it does not belong in the code.
- **Every output path comes from `config.yaml` too.** `paths.artifacts` is the root and
  `outputs` names each directory and fixed file; code resolves them through
  `config.output_path(config, key)` rather than writing a literal. Point `paths.artifacts`
  somewhere else and the whole run follows. The only paths composed in code are leaves the
  code generates per item, such as `task_<i>.json` and `panel_<name>.csv`, inside a
  directory resolved from config.
- **Seeds are explicit.** `random`, `numpy` and the framework RNG are all set at startup
  from `config.yaml`'s `seed`.
- **Tests come first**, and they pin behaviour at named seams rather than restating the
  implementation.
- **Inferences are labelled as inferences.** Where the paper is silent or ambiguous, the
  code and `config.yaml` say so at the point of use instead of quietly choosing.
- **`artifacts/` is the single output location**, and it is configurable. Nothing is
  written outside it.
