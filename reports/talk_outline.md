# Talk outline — The Asynchronous State in Cortical Circuits (+ LSM extension)

About 20 minutes, 14 slides. Figures are in `reports/figures/`. The numbers and caveats
behind every claim are in `renart2010_reproduction.md` and `lsm_technical.md`.

---

**1. Title** — Renart et al., *Science* 2010. Reproduction of Figs. 1–3, plus an extension
to liquid state machines.

**2. The puzzle (1 min).** Cortical cells share 10–30 % of their inputs, yet recordings
(Ecker et al. 2010, this paper's in vivo data) show near-zero mean pairwise correlations.
How can dense networks be asynchronous? Shared input should force correlations of order p.

**3. Fig. 1B/1E — the arithmetic (2 min).** `fig1b.png`, `fig1e.png`
- With E-only inputs, current correlation c = p exactly (0.20 at p = 0.2), and correlated
  input drives the output to r ≈ 0.93.
- Add correlated inhibition and c stays 0.20 → 0.26: the E–I cross term is negative and cancels.
- Message: correlations cancel *if* E and I fluctuations track each other.

**4. Fig. 2B/2D — balance and tracking in a recurrent network (2 min).** `fig2b.png`, `fig2d.png`
- E and I inputs are each 10–20 units from threshold; their sum hovers just below it.
- Inhibitory population activity follows excitatory activity, which follows the input.

**5. Fig. 2E/2C — the cancellation and the 1/N law (2 min).** `fig2e.png`, `fig2c.png`
- At N = 8192: II +0.25, EE +0.11, XX +0.04, EI −0.16 → total +0.017.
- r̄ falls as 1/N (2.0e-2 → 6.9e-4 over N = 100 → 3160), c as ~1/√N; components stay O(1).
- Say: "nothing is fine-tuned; the cancellation is dynamical."

**6. Fig. 3A/3B — a realistic conductance-based network (2 min).** `fig3a.png`, `fig3b_2000s.png`
- 4000 E + 1000 I + 4000 Poisson inputs, p = 0.2, SOM parameters.
- 10 networks × 200 s: E 1.04 Hz, I 3.98 Hz (paper 1.0 / 3.6), r̄ = 1.6 × 10⁻⁴ (paper < 10⁻³).
- 3B: 499 500 pairs, centred on 0 and wider than the jittered null (σ 0.0054 vs 0.0041).

**7. Fig. 3C/3D — the testable prediction (2 min).** `fig3c.png`, `fig3d.png`
- Voltage CCGs with spiking off: EPSP–EPSP +0.25, IPSP–IPSP +0.16, **EPSP–IPSP −0.19**,
  rest +0.03. The V-shape against holding potential.
- Say: "you can test this with paired whole-cell recordings at two holding potentials."

**8. What I had to do to reproduce it (1 min).**
- Synapses integrated per neuron instead of per synapse. This is exact, because the kinetics
  are linear. A 200 s network now takes 12 min instead of 27 h, so every Fig. 3 protocol ran
  at the paper's length.
- Deviations: Fig. 2C used a reduced sweep (40k τ); I rate is +10 %; and a few SOM ambiguities
  are listed in the report.

**9. Extension: does this hold in a liquid state machine? (1 min)**
- Questions: (i) is the claim valid at reservoir size, (ii) what makes an LSM *faithful* to
  the argument, (iii) is a faithful LSM a better liquid on N-MNIST / DVS Gesture?
- N = 500 (400 E / 100 I), the smallest N with √K ≈ 9 ≫ 1.

**10. What "faithful" requires (1.5 min).**
- Four ingredients: dense p, strong coupling (J ~ 1/√K), fast and strong inhibition under
  Dale's law, and shared input.
- The key adjustment: g_ab(N) = g_ab^paper · √(N_b^paper / N_b). Unscaled at N = 500 the
  network is **silent** (0 Hz). Scaled, it runs at **1.26 / 3.85 Hz**, the paper's operating
  point, with no tuning.
- Comparators: dense_weak (dense wiring but unscaled coupling) and a Maass 2002 column
  (λ = 2, in-degree ~6). One calibrated gain each, so every comparison is at equal rate.

**11. Result 1 — yes, the asynchronous state survives (2 min).** `lsm_scaling.png`, `lsm_current_scaling_T.png`
- Faithful family N = 250 → 5000: **r̄ ∝ N^−1.8**, c_T ∝ N^−0.66; at N = 5000 it *is* block 3.
- At N = 500 and equal rate: r̄ is 0.012 (faithful), 0.019 (dense_weak) and **0.039 (Maass)**.
- `lsm_currents_T.png`: faithful cancels +1.03 of positive terms down to 0.05; the Maass
  liquid has nothing to cancel (c ≈ c_XX ≈ 0.15).

**12. Result 2 — but it is not a better liquid (2 min).** `lsm_readout.png`
- N-MNIST: faithful **91.3 %**, against input only 96.0 %, dense_weak 97.0 % and Maass 97.1 %
  (±1.2).
- DVS Gesture: 53.8 / 57.6 / 60.6 / 60.6 % — all within ±6, not separable on 264 test
  samples.

**13. Why (2 min).** `lsm_reliability.png`
- Present the same sample twice with independent noise: only **28 %** of the faithful
  liquid's state variance repeats on N-MNIST (66 % on Gesture), against 88–99 % for the Maass
  liquid.
- The strong coupling that cancels correlations also makes the network generate its own
  irregular activity. That activity decorrelates neurons from each other *and* from trial to
  trial: high dimensionality (participation ratio 179 against 10), mostly noise.
- N = 2000 test: see `lsm_technical.md` §6.3 — does averaging over more neurons recover it?

**14. Take-home (1 min).**
- The paper's mechanism reproduces quantitatively, at every level of the argument.
- An LSM can be made faithful to it, and it then becomes asynchronous by cancellation. This
  requires √K-scaled strong coupling.
- For computation, asynchrony through chaotic balance trades reliability for decorrelation.
  A good liquid needs input-locked diversity, not decorrelation as such.
- Next: balanced networks in a less chaotic regime (weaker coupling, or slower inhibition), or
  readouts that average over population noise.
