# The asynchronous state in a liquid state machine — technical report

Extension of Renart et al. (2010) to reservoir computing. Code: `src/lsm/` (entry point
`python -m lsm.run`), configuration: `config.yaml → lsm`, cluster scripts:
`slurm/lsm_*.slurm`. Every number below comes from runs of that code at the commits on `main`.

**Questions**

1. Does the paper's claim (dense, strongly coupled E/I networks are asynchronous because
   inhibition tracks and cancels shared excitation, so `r̄ ~ 1/N`) hold in a
   **reservoir-sized** spiking network (~500 neurons)?
2. What must be adjusted to make an LSM **faithful** to the paper's argument?
3. Is the faithful LSM a **better liquid** for event-camera classification (N-MNIST, DVS
   Gesture) than a classic Maass-style liquid?

## 1. What the paper's argument requires

The SOM's theory (binary network, S§2.1–2.2) and the spiking network (S-p.20–21) rest on
four ingredients. An LSM is faithful to the argument only if it keeps all four.

| # | Ingredient | Where it enters the argument | Typical LSM (Maass et al. 2002) |
|---|---|---|---|
| 1 | **Dense** connectivity: `p` fixed (0.2) as N grows, so every pair shares a fraction p of its inputs | creates the shared-input correlation that must be cancelled | sparse, local (λ = 2 grid units), in-degree ~5–10 |
| 2 | **Strong** coupling: `J ~ 1/√K`, so mean E and I inputs are `O(√K)` and cancel to `O(1)` (balance) | balance is what lets I track E at a fast time scale | weights fixed, not scaled with K |
| 3 | **Dale's law with fast, strong inhibition** (`g_EI = g_II = 40 nS` against `g_EE = 2.4 nS`; I delays 0.1–0.9 ms against 0.5–1.5 ms for E) | the tracking condition: inhibition must follow excitation within a few ms | E/I mixed, inhibition not privileged |
| 4 | **Shared external drive X** with `p_X = p` | the correlated input whose fluctuations are tracked and cancelled | input to a random subset, usually 30 % of cells |

## 2. Size of the reservoir

The paper's spiking network has N = 5000 (4000 E + 1000 I); the binary network is swept over
N = 100 … 8192. **We use N = 500 (400 E + 100 I, the 4:1 ratio of both the paper and Maass
2002)**. That is the smallest size at which ingredient 2 still has a meaning:

- with `p = 0.2`, the E in-degree is `K_EE = 80`, and the balance parameter `√K ≈ 9 ≫ 1`.
  At N = 100 it would be `√16 = 4`, and the O(1) residual would be comparable to the O(√K)
  terms that are supposed to cancel;
- the theory predicts `r̄ ~ 1/N`, so N = 500 is 10× more correlated than the paper's
  network. That is a *testable* prediction, not a defect;
- N = 562 is a point on Fig. 2C's own grid, and LSMs for event data are typically 10²–10³
  neurons.

The asynchrony test spans **N = 250, 500, 1000, 2000, 5000**, so the reservoir size sits
inside a scaling law whose N = 5000 end **is the paper's network**.

## 3. Adjustments needed for a faithful LSM

### 3.1 Coupling must be rescaled — the network is silent otherwise

Shrinking block 3 to N = 500 with the paper's conductances (fixed nS values) gives **0 Hz**:
each neuron loses 90 % of its inputs, so the mean drive falls 10× and nothing reaches threshold.
The theory says how to shrink a balanced network: keep `J√K` fixed. For conductances, per
presynaptic population b:

```
g_ab(N) = g_ab^paper · √( N_b^paper / N_b )          (b ∈ {E, I, X})
```

This keeps `K_b · g_ab²` (the input variance) fixed while the mean E and I inputs scale as
`√K`. At N = 5000 it reduces exactly to the paper's values: a test pins this. Measured at
N = 500 with **no tuning**: **E 1.26 Hz, I 3.85 Hz** (paper at N = 5000: 1.0 / 3.6 Hz).

| N = 500 network | E rate | I rate |
|---|---|---|
| paper conductances, unchanged | 0.00 Hz | 0.00 Hz |
| √K-scaled (faithful) | 1.26 Hz | 3.85 Hz |

### 3.2 Everything else is kept

- Bernoulli(p = 0.2) wiring for all 6 population pairs, self-connections excluded.
- Dale's law, conductance-based synapses, `V_rev^I = −80 mV`, and the faster inhibitory
  delays.
- Per-synapse heterogeneity: g ~ truncated Gaussian(mean g, sd 0.5 g), with corrected
  moments.
- `dt = 0.05 ms`, rk2, and the same neuron (C_m = 0.25 nF, g_L = 16.7 nS, θ = −50 mV,
  reset −60 mV, refractory 2/1 ms).
- The 2.5 Hz Poisson background stays on every input channel, so the reservoir sits in its
  asynchronous operating point before, between and during stimuli.

### 3.3 Input encoding for event cameras (needed by any LSM, faithful or not)

- One X neuron per pooled pixel and polarity: N-MNIST 34×34 → 17×17×2 = **578** channels;
  DVS Gesture 128×128 → 16×16×2 = **512** channels. N_X therefore changes with the task, and
  the X conductances follow the same √ rule (`g_aX ∝ √(4000/N_X)`).
- Events are **thinned** with one dataset-wide keep-probability, which brings the mean
  stimulus rate per channel to 20 Hz (8× the background), inside the balanced network's
  linear-response range. q is estimated on 500 training samples only (N-MNIST q ≈ 0.7–0.8,
  DVS Gesture q = 0.086).
- Samples are laid end to end with a 50 ms gap (~3 τ_m) so their states do not carry over.

### 3.4 Implementation: no second simulator

All variants are `block3.model.build_network` with different parameters. Two small,
test-pinned extensions made this possible:

- the per-neuron synaptic integration: exact, and ~140× faster;
- a `connect` hook for the Maass column topology.

The network an LSM result is reported on is therefore the same code that reproduces Fig. 3.

## 4. The variants compared

| variant | wiring | recurrent coupling | free parameter |
|---|---|---|---|
| **faithful** | Bernoulli p = 0.2 (dense) | √K-scaled (strong) | none |
| **dense_weak** (ablation) | Bernoulli p = 0.2 (dense) | paper nS values (weak at this K) | input gain |
| **maass** (classic LSM) | 3-D column, 5×5 cross-section, `P = C_ab·exp(−(D/λ)²)`, λ = 2, C = 0.3 / 0.2 / 0.4 / 0.1 for EE / E→I / I→E / II | paper nS values | input gain |

All three share the same input pathway (X → reservoir Bernoulli p = 0.2, √-scaled g). The one
free gain of each non-faithful variant is calibrated by bisection so that comparisons are at
**matched firing rate**:

- *asynchrony runs* match the background E rate (1.41 Hz at N = 500): gains dense_weak 0.952,
  maass 0.912;
- *classification runs* match the stimulus-driven rate (34.9 Hz on DVS Gesture training
  samples): gains dense_weak 0.559, maass 0.216.

Matching at background alone is not enough. The same stimuli then drive faithful at 36 Hz,
dense_weak at 61 Hz and maass at **206 Hz**. The unbalanced liquids have a much steeper gain,
and the Maass liquid needs 4.6× less input than faithful for the same driven rate. That is
itself a signature of the paper's mechanism: tracking makes the balanced network's response
linear and moderate.

## 5. Results — asynchrony (question 1)

*(filled in from `artifacts/lsm/asynchrony.csv`; see figures)*

## 6. Results — classification (question 3)

*(filled in from `artifacts/lsm/readout.csv`)*

## 7. Reproduce

```bash
PYTHONPATH=src python -m lsm.run calibrate                 # background gains (5 min)
PYTHONPATH=src python -m lsm.run calibrate_task dvs_gesture
sbatch slurm/lsm_asynchrony.slurm                          # 21 tasks
python src/lsm/prepare_nmnist.py <tonic dir> <out dir>      # once, needs tonic
for v in input faithful dense_weak maass; do
  sbatch --array=0-13 slurm/lsm_features.slurm dvs_gesture $v
  sbatch --array=0-23 slurm/lsm_features.slurm nmnist $v
done
PYTHONPATH=src python -m lsm.run aggregate
PYTHONPATH=src python -m lsm.run readout dvs_gesture && PYTHONPATH=src python -m lsm.run readout nmnist
PYTHONPATH=src python -m lsm.run plot
```
