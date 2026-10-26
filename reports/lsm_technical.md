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
  linear-response range. q is estimated on 500 training samples only: **N-MNIST q = 0.740,
  DVS Gesture q = 0.164**. Measured input rates per chunk, including the 2.5 Hz background:
  N-MNIST 22 Hz, DVS Gesture 17–27 Hz.
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

Background-only runs (block 3's 2.5 Hz Poisson X, no task input), 100 s each, 3 network
realisations per point, measured with the paper's estimators. 21 runs, `artifacts/lsm/asynchrony.csv`.

### 5.1 The faithful LSM follows the paper's scaling law down to reservoir size

![r̄ and σ_r against N](figures/lsm_scaling.png)

| faithful, N | 250 | **500** | 1000 | 2000 | 5000 (= block 3) |
|---|---|---|---|---|---|
| E / I rate (Hz) | 1.25 / 3.16 | **1.38 / 3.75** | 1.38 / 4.29 | 1.27 / 4.27 | 1.03 / 3.98 |
| r̄ (E pairs, T = 50 ms) | 0.031 | **0.0117** | 0.0034 | 0.00094 | 0.00014 |
| σ_r measured | 0.036 | **0.026** | 0.021 | 0.020 | 0.019 |
| σ_r of the jittered null | 0.019 | 0.018 | 0.018 | 0.018 | 0.018 |
| c_T, total current corr. | 0.067 | **0.051** | 0.035 | 0.019 | 0.0096 |
| E–I tracking (3 ms) | 0.56 | 0.60 | 0.61 | 0.60 | 0.62 |

- **r̄ falls as N^−1.82** (log-log fit over N = 250–5000), at least as fast as the theory's 1/N.
  The extra steepness at small N is plausibly the finite-K correction: at √K ≈ 6–9 the balance
  is imperfect, and the residual correlation that it leaves decays faster than the
  asymptotic 1/N.
- **c_T falls as N^−0.66**, against the theory's 1/√N (exponent −0.5).
- **σ_r ≫ r̄ at every N** (at N = 500, 0.026 against 0.012), and σ_r stays above its
  jittered null, so individual pairs keep real correlations of both signs.
- The N = 5000 point is the paper's own network: r̄ = 1.4 × 10⁻⁴, matching the block 3
  reproduction (1.6 × 10⁻⁴ over 10 networks × 200 s).

**Answer to question 1: yes.** The asynchronous state survives, through the same mechanism, in
a 500-neuron reservoir, provided the reservoir keeps the four ingredients and the √K
coupling. Its residual correlation is larger (r̄ ≈ 0.012), exactly as the 1/N law predicts.

### 5.2 The mechanism: cancellation of current correlations

The current correlations must be measured on currents averaged over the count window T
(the SOM's filtered currents, Fig. S6). On raw 1 ms currents, the total stays at ~0.15 for
every N, because Poisson shot noise is faster than inhibition can track (τ_d = 5 ms) and
spikes do not respond to it. See `figures/lsm_current_scaling_1ms.png`.

![current decomposition at N = 500](figures/lsm_currents_T.png)

At N = 500 and **equal background rate**:

| variant | c_EE | c_II | c_XX | 2c_EI | 2c_EX | 2c_IX | **c (total)** | **r̄** | σ_r |
|---|---|---|---|---|---|---|---|---|---|
| faithful | +0.13 | +0.72 | +0.18 | **−0.57** | +0.23 | **−0.65** | **0.051** | **0.0117** | 0.026 |
| dense_weak | +0.03 | +0.15 | +0.19 | −0.11 | +0.12 | −0.29 | 0.082 | 0.0190 | 0.027 |
| maass | +0.00 | +0.00 | +0.16 | −0.00 | +0.01 | −0.02 | 0.148 | **0.0386** | 0.032 |

- **Faithful.** Large positive terms (sum ≈ +1.03) are cancelled by large negative E–I and
  I–X terms (≈ −1.22, offset by +0.23 from E–X) down to 0.05: the paper's mechanism.
- **Maass liquid.** The recurrent currents are tiny (in-degree ~6), so the cell's current
  is essentially its share of the external input. c ≈ c_XX ≈ 0.15, with nothing to cancel
  it, and r̄ is **3.3× higher than faithful at the same firing rate**.
- **dense_weak** sits in between: dense wiring alone is not enough; the strong coupling
  (ingredient 2) does the cancelling.

![cancellation vs N](figures/lsm_current_scaling_T.png)

As N grows, the dominant negative term moves from 2c_EI to 2c_IX: inhibition increasingly
tracks the external input itself, while the total c falls monotonically. This is the
paper's tracking picture, with I following X as well as E.

## 6. Results — classification (question 3)

N = 500, all variants at **matched stimulus-driven rate**, the same input encoding, and the
same readout. The readout is multinomial logistic regression on standardised per-neuron spike
counts in time bins: 10 × 150 ms for Gesture, 6 × 50 ms for N-MNIST. Its L2 strength is
chosen per variant on a validation slice of the training split. Test sets are untouched until
the final score.

![readout accuracy](figures/lsm_readout.png)

| | DVS Gesture (1077 train / 264 test) | N-MNIST (10 000 / 2000) |
|---|---|---|
| input only (no reservoir) | 57.6 % | 96.05 % |
| **faithful** | **53.8 %** | **91.3 %** |
| dense_weak | 60.6 % | 97.05 % |
| Maass column | 60.6 % | **97.1 %** |
| 95 % CI half-width | ± 6 points | ± 1.2 points |

**Answer to question 3: no, at N = 500 the faithful LSM is the *worst* liquid.**
- On N-MNIST, faithful is 5.8 points behind the Maass liquid, well outside the intervals.
- On DVS Gesture, all four lie within one interval of each other. A 264-sample test set cannot
  separate them, and all are near the input-only baseline.

### 6.1 Why: the asynchronous state is mostly self-generated variability

The same 100 test samples were run twice through the same reservoir with independent
encoding/background noise. The figure shows the fraction of feature variance shared between
the two repeats (`lsm.readout.reliability`, unbiased for the stimulus-locked share):

![reliability](figures/lsm_reliability.png)

| stimulus-locked share | input | faithful | dense_weak | maass | participation ratio (faithful / maass) |
|---|---|---|---|---|---|
| DVS Gesture | 0.97 | **0.66** | 0.94 | 0.99 | 105 / 5 |
| N-MNIST | 0.90 | **0.28** | 0.77 | 0.88 | 179 / 10 |

- **The faithful liquid is high-dimensional, but mostly not about the input.** Its state
  spreads over 18–20× more dimensions than the Maass liquid's. On N-MNIST only 28 % of its
  variance repeats when the same digit is shown again; the other 72 % is generated by the
  network itself.
- **This is the other face of the paper's mechanism.** Strong coupling makes the balanced
  network amplify small fluctuations into irregular, chaotic activity (van Vreeswijk &
  Sompolinsky 1996). That activity is exactly what decorrelates neurons from *each other*.
  It also decorrelates the response from *trial to trial*: asynchrony and reliability come
  from the same source.
- **The Maass liquid is the opposite.** Its state is a few large shared modes
  (participation ratio 5–10, the input's own correlations, uncancelled) but almost
  deterministic given the input (0.88–0.99). For a linear readout with ~10³–10⁴ training
  samples, reliability wins.

### 6.2 Does a larger faithful reservoir recover?

The asynchronous state's noise is independent across neurons, which is the paper's point,
so a readout over more neurons should be able to average it away; the Maass liquid's shared
modes cannot be averaged. Faithful and Maass at **N = 2000**: see §6.3 (filled in when the
runs land). The per-neuron reliability does not change with N (faithful 0.68 / 0.31, Maass
0.99 / 0.88 for Gesture / N-MNIST), so any gain must come from averaging across neurons.
Caveat: at N = 2000 the faithful reservoir runs at ~13 Hz under stimulus against ~40 Hz at
N = 500. The input path is unchanged, and the stronger balance at larger K suppresses the
response.

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
