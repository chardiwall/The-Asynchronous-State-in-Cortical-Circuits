# Block 3 — Recurrent Conductance-Based Spiking Network

Implementation notes for `docs/paper/03-recurrent-spiking-network.md` (Fig. 3) — this repo's
**main reproduction target** (L1) and the network that L2 replaces with a spiking reservoir /
liquid state machine. This file documents the *implementation*; the paper extraction itself
lives in `docs/paper/03-recurrent-spiking-network.md` and stays paper-only, no code detail.

## Architecture schematic

Three populations, densely and independently connected (`p = 0.2` for every ordered pair),
Dale's-law-respecting (one sign per presynaptic population):

```
                 p=0.2, g^EX=5.4nS                 p=0.2, g^IX=5.4nS
   X (N=4000) ───────────────────────┐   ┌───────────────────────── X (N=4000)
   Poisson, 2.5 Hz, no recurrent      │   │
   input, feeds E and I only         ▼   ▼
                              ┌─────────────┐
                p=0.2         │             │        p=0.2
     ┌───────── g^EE=2.4nS ───│   E (4000)  │─── g^EI=40nS ─────────┐
     │                        │             │                       │
     │            ┌───────────┴─────────────┴───────────┐           │
     │            │                                      │           │
     ▼            │              p=0.2, g^IE=4.8nS       ▼           ▼
┌─────────┐       └──────────────────────────────────►┌─────────┐
│  E (4000) │◄──────────── p=0.2, g^II=40nS ───────────│  I (1000) │
└─────────┘                                            └─────────┘
```

(Read as: every ordered population pair `(post ← pre)` has an independent `p=0.2` Bernoulli
connection and its own mean conductance `g^{post,pre}`; the diagram collapses the usual
four-quadrant E/I wiring diagram into one box per population to keep it legible.)

**Per neuron** (population `α ∈ {E, I}`; `X` has no recurrent input, no membrane equation —
it is a pure Poisson spike source):

```
C_m dV_i^α/dt = −g_L(V_i^α − V_L) + I_i^αE(t) + I_i^αI(t) + I_i^αX(t) + I_i^app   (V_i^α < θ)
```
spike at `θ = −50 mV` → reset to `V_R = −60 mV`, hold `t_ref` (2 ms E, 1 ms I).

**Per synapse** `(i ← j)`, population pair `(α ← β)`:

```
I_i^αβ(t)      = −[ p_ij^αβ · g_ij^αβ · s_ij^αβ(t) ] · (V_i^α − V_rev^β)
τ_d ds_ij/dt   = x_ij − s_ij
τ_r dx_ij/dt   = τ̃ Σ_spikes δ(t − t_j − d_ij) − x_ij
```
`g_ij^αβ ~ Gaussian(g^αβ, 0.5·g^αβ)`, resampled if negative (`docs/adr/0004`); delay `d_ij`
uniform per synapse (`[0.5,1.5]ms` from E, `[0.1,0.9]ms` from I), 0.05 ms resolution. `V_rev^E =
V_rev^X = 0 mV`, `V_rev^I = −80 mV` — inhibition is both stronger (`g^EI=g^II=40nS` vs.
`g^EE=2.4nS`, `g^IE=4.8nS`) and faster (shorter delay) than excitation, the precondition for
**tracking** (see `CONTEXT.md`).

The measurement applied on top (unchanged from blocks 1/2, `src/analysis.py`): population
firing rates, spike-count correlation `r̄`/`σ_r` (`T=50ms`), current-component correlations —
the same estimators at every level L1–L4.

## Architecture comparison: this network vs. a reservoir / liquid state machine (L2)

| | **Block 3 (this network, L1)** | **Spiking reservoir / LSM (L2 target)** |
|---|---|---|
| Connectivity | Dense, fixed `p=0.2` for every one of the 9 ordered population pairs; explicitly structured to match block 2's binary network | Typically sparse/random, no population-pair structure prescribed — a generic recurrent graph |
| Dale's law | Enforced by construction: one sign per presynaptic population (E excitatory, I inhibitory, X excitatory) | Not required; an LSM's recurrent weights are usually signed per-synapse with no population-level sign constraint |
| Weight heterogeneity | Per-synapse Gaussian around population-pair-specific means (`g^EE`, `g^EI`, …) — 6 distinct means | Usually one random-weight distribution for the whole reservoir, or a spectral-radius-controlled random matrix — no population-specific tuning |
| What the weights encode | Fitted/assumed to produce a specific studied phenomenon: **tracking** (`m_E(t)=A_E m_X(t)`) and the resulting `c_EI` cancellation | Fixed *not* to reproduce a specific target dynamic — the reservoir's only job is to be a rich, input-driven dynamical substrate |
| Readout | None — the network's own population activity *is* the object of study (no downstream task) | A trained (linear, or otherwise simple) readout layer maps reservoir state to a task output — central to the LSM's purpose |
| Is the asynchronous state a design goal? | Yes — it is the paper's claim, and this network is built (dense + strongly coupled) specifically so it emerges dynamically | No — asynchrony is not designed for; L2 asks whether it **survives** the substitution anyway |
| Training / plasticity | None; all weights fixed at construction | Recurrent weights fixed (as in block 3), but the *readout* is trained — the one place learning happens in an LSM |
| Role in this repo | L1: the thing being reproduced | L2: the thing block 3's network is swapped out for, keeping block 3's `X` population's Poisson drive and the shared analysis pipeline (`r̄`, `σ_r`, current correlations) fixed, so any change in the measurement is attributable to the network substitution alone |

The comparison matters for L2's question ("does the asynchronous state survive a network
substitution?") precisely because the two architectures share almost nothing structurally
(no Dale's law, no population-pair-specific weights, no design intent around tracking) — an LSM
that still produces `r̄ ~ 1/N` would show the asynchronous state is a broader dynamical
phenomenon, not an artifact of block 3's specific connectivity choices.

## Phase roadmap

- **Phase 0** — ADR (conductance sign handling) + config + test-first model build. *(this
  session)*
- **Phase 1** — Local exploratory pass: full paper-scale `N`, `length_s=5`, single network,
  Brian2 runtime mode — sanity-check only (NaN-free, non-zero, non-runaway rates). *(this
  session)*
- **Phase 2** — Fig. 3A–B full-scale reproduction (`length_s=200`, 5–10 networks) against the
  paper's targets (E≈1 Hz, I≈3.6 Hz, `r̄<0.001`) — likely needs the DGX/Slurm given ~9M
  synapses; sizing decided against a measured throughput benchmark, not guessed.
- **Phase 3** — Fig. S6 detailed characterisation (single network, 5000 s; `r̄` vs. count
  window `T`; filtered-current correlation vs. `τ_f`).
- **Phase 4** — Fig. S7 robustness sweeps (`ν_X` 0→40 Hz, `p` 0→0.4, `τ_E` 5→2 ms) — the
  `p`-sweep is this repo's sharpest single check that the mechanism (not shared-input fraction)
  sets `r̄`.
- **Phase 5** — Fig. S8 non-stationary sinusoidal drive (`ν_X(t)`, shift-predictor CCG
  correction).
- **Phase 6** — Fig. 3C–D intracellular cancellation protocol (spiking disabled, `I_app` swept,
  membrane-potential CCGs across 450–1000 pairs, 10 networks).

Each phase is independently completable and checkable against the paper's own stated target for
that figure/panel (`docs/paper/06-figure-protocols.md`); later phases reuse Phase 0's `model.py`
unchanged except for the specific protocol modification each panel needs (e.g. Phase 6 disables
the threshold/reset mechanism for one recorded pair only).
