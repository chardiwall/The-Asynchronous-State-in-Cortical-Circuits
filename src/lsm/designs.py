"""Three PUBLISHED liquids (config lsm.designs): Maass, Natschlaeger & Markram 2002; Kaiser
et al. 2017; Wijesinghe et al. 2019 -- the trio compared by Woo et al. 2024 (Table 1).

build_design returns block3.model's Block3Network, and each neuron carries the same
per-channel input split I_E, I_I, I_X, so lsm.asynchrony and lsm.reservoir measure these
liquids exactly as they measure the Renart-derived variants.

  current      tau_m dV/dt = -(V - V_L) + R_m (I_E + I_I + I_X + I_back); each I_c an
               exponential PSC (tau by presynaptic type, X excitatory), jumping by w (Maass,
               Kaiser). Linear, so integrated exactly.
  conductance  tau_m dV/dt = -(V - V_L) + R_m (g_E (E_exc - V) + g_I (E_inh - V) + g_X (E_exc - V)),
               each g_c exponential, jumping by w (Wijesinghe, after Diehl & Cook 2015).
  STP          Markram et al. 1998 / NEST tsodyks_synapse: between spikes u -> 0 (tau F) and
               x -> 1 (tau D); at a spike u += U(1 - u), the PSC jumps by w u x, then x -= u x.
               Every STP synapse starts at rest (u = 0, x = 1); runs discard a burn-in.
"""
import brian2 as b2
import numpy as np

from block3.connectivity import generate_bernoulli_connectivity
from block3.model import Block3Network
from lsm.variants import column_connect, column_positions

CURRENT_EQS = """
dV/dt = (-(V - V_L) + R_m * (I_E + I_I + I_X + I_back)) / tau_m : volt (unless refractory)
dI_E/dt = -I_E / tau_E : amp
dI_I/dt = -I_I / tau_I : amp
dI_X/dt = -I_X / tau_E : amp
"""
CONDUCTANCE_EQS = """
dV/dt = (-(V - V_L) + R_m * (I_E + I_I + I_X)) / tau_m : volt (unless refractory)
I_E = g_E * (E_exc - V) : amp
I_I = g_I * (E_inh - V) : amp
I_X = g_X * (E_exc - V) : amp
dg_E/dt = -g_E / tau_E : siemens
dg_I/dt = -g_I / tau_I : siemens
dg_X/dt = -g_X / tau_E : siemens
"""
COLUMN_NAME = {"E_to_E": "EE", "E_to_I": "IE", "I_to_E": "EI", "I_to_I": "II"}  # post<-pre


def _redraw(sample, ok, n: int) -> np.ndarray:
    """NEST's clipped distributions: draw, then redraw every value outside the support."""
    x = sample(n)
    bad = ~ok(x)
    while bad.any():
        x[bad] = sample(int(bad.sum()))
        bad = ~ok(x)
    return x


def draw_weights(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    (kind, p), = spec.items()
    if kind == "uniform":
        return rng.uniform(p["low"], p["high"], n)
    mean, sd = p["mean"], p["cv"] * abs(p["mean"])
    if kind == "normal":
        return rng.normal(mean, sd, n)
    if kind == "gamma":   # Maass: SD = cv * |mean|, gamma-distributed, sign from the mean
        shape = 1.0 / p["cv"] ** 2
        return np.sign(mean) * rng.gamma(shape, abs(mean) / shape, n)
    if kind == "normal_clipped":   # NEST: redrawn until on the mean's side of 0
        return _redraw(lambda k: rng.normal(mean, sd, k), lambda x: x * np.sign(mean) >= 0, n)
    raise ValueError(f"unknown weight distribution {kind}")


def _delays_ms(design: dict, spec: dict, n: int, rng) -> np.ndarray:
    if "delay" in spec:
        return np.full(n, float(spec["delay"]))
    if "delay" in design:
        d = design["delay"]
        return _redraw(lambda k: rng.normal(d["mean"], d["sd"], k),
                       lambda x: (x >= d["low"]) & (x <= d["high"]), n)
    return np.zeros(n)


def _stp_values(design: dict, stp: dict, n: int, rng) -> dict:
    """Maass: Gaussian with SD stp_cv * mean, negatives redrawn from U(0, 2 mean)."""
    cv = design.get("stp_cv", 0.0)
    values = {}
    for key, mean in stp.items():
        x = rng.normal(mean, cv * mean, n) if cv > 0 else np.full(n, float(mean))
        bad = x <= 0
        x[bad] = rng.uniform(0, 2 * mean, int(bad.sum()))
        values[key] = x
    return values


def _edges(pair: str, spec: dict, sizes: dict, rng, column) -> tuple:
    pre, post = pair.split("_to_")
    n_pre, n_post = sizes[pre], sizes[post]
    if "p" in spec:
        return generate_bernoulli_connectivity(n_pre, n_post, spec["p"], rng, exclude_self=pre == post)
    if "k" in spec:   # NEST fixed_indegree, autapses and multapses allowed
        return rng.integers(0, n_pre, n_post * spec["k"]), np.repeat(np.arange(n_post), spec["k"])
    if "outdegree" in spec:   # NEST fixed_outdegree, likewise
        k = spec["outdegree"]
        return np.repeat(np.arange(n_pre), k), rng.integers(0, n_post, n_pre * k)
    return column(COLUMN_NAME[pair], n_pre, n_post, rng, pre == post)


def static_synapses(source, target, var, edges, w, delay_ms, name):
    unit = "amp" if w.has_same_dimensions(b2.amp) else "siemens"
    syn = b2.Synapses(source, target, f"w : {unit}", on_pre=f"{var}_post += w", name=name)
    if len(edges[0]):
        syn.connect(i=np.asarray(edges[0]), j=np.asarray(edges[1]))
        syn.w = w
        syn.delay = delay_ms * b2.ms
    return syn


def stp_synapses(source, target, var, edges, w, stp, delay_ms, name):
    unit = "amp" if w.has_same_dimensions(b2.amp) else "siemens"
    model = f"""w : {unit}
                U : 1
                tau_d : second
                tau_f : second
                du/dt = -u / tau_f : 1 (event-driven)
                dx/dt = (1 - x) / tau_d : 1 (event-driven)"""
    syn = b2.Synapses(source, target, model, name=name,
                      on_pre=f"u += U * (1 - u)\n{var}_post += w * u * x\nx -= u * x")
    if len(edges[0]):
        syn.connect(i=np.asarray(edges[0]), j=np.asarray(edges[1]))
        syn.w, syn.U = w, stp["U"]
        syn.tau_d, syn.tau_f = stp["D"] * b2.ms, stp["F"] * b2.ms
        syn.u, syn.x = 0.0, 1.0
        syn.delay = delay_ms * b2.ms
    return syn


def _neuron_group(design: dict, pop: str, n: int, rng) -> b2.NeuronGroup:
    p, tau = design[pop], design["tau_syn_ms"]
    current = design["neuron"] == "current"
    namespace = {"tau_m": p["tau_m"] * b2.ms, "R_m": p["r_m"] * b2.Mohm, "V_L": p["v_leak"] * b2.mV,
                 "theta": p["v_threshold"] * b2.mV, "v_reset": p["v_reset"] * b2.mV,
                 "tau_E": tau["E"] * b2.ms, "tau_I": tau["I"] * b2.ms}
    if current:
        namespace["I_back"] = p["i_back_nA"] * b2.nA
    else:
        namespace.update(E_exc=p["e_exc"] * b2.mV, E_inh=p["e_inh"] * b2.mV)
    group = b2.NeuronGroup(n, CURRENT_EQS if current else CONDUCTANCE_EQS, threshold="V > theta",
                           reset="V = v_reset", refractory=p["t_ref"] * b2.ms, namespace=namespace,
                           method="exact" if current else "exponential_euler", name=f"group_{pop.lower()}")
    low, high = design["initial_v"] or (p["v_leak"], p["v_leak"])
    group.V = rng.uniform(low, high, n) * b2.mV
    return group


def build_design(config: dict, name: str, n_external: int, external: dict,
                 rng: np.random.Generator, gain_key: str = "task_input_gain") -> Block3Network:
    """Design `name` at N = lsm.size, driven by `external` (block 3's Poisson X or explicit
    spike_generator events), its X weights times lsm[gain_key][name]."""
    lsm, design = config["lsm"], config["lsm"]["designs"][name]
    b2.start_scope()
    b2.defaultclock.dt = config["spiking_network"]["simulation"]["dt_ms"] * b2.ms
    b2.seed(int(rng.integers(2 ** 31 - 1)))   # Brian2's own Poisson draws, from the run's seed
    n_e = round(lsm["excitatory_fraction"] * lsm["size"])
    sizes = {"E": n_e, "I": lsm["size"] - n_e, "X": n_external}
    groups = {"E": _neuron_group(design, "E", sizes["E"], rng), "I": _neuron_group(design, "I", sizes["I"], rng)}
    if external["kind"] == "spike_generator":
        groups["X"] = b2.SpikeGeneratorGroup(n_external, external["indices"],
                                             np.asarray(external["times_ms"]) * b2.ms, name="group_x")
    else:
        groups["X"] = b2.PoissonGroup(n_external, rates=external["rate_hz"] * b2.Hz, name="group_x")

    column = column_connect(config, column_positions(config, sizes["E"], sizes["I"], rng), 0.0)
    unit = b2.nA if design["neuron"] == "current" else b2.nS
    prefix = "I" if design["neuron"] == "current" else "g"
    synapses = {}
    for pair, spec in design["pairs"].items():
        pre = pair.split("_to_")[0]
        edges = _edges(pair, spec, sizes, rng, column)
        n = len(edges[0])
        w = draw_weights(spec["weight"], n, rng) * (lsm[gain_key][name] if pre == "X" else 1.0) * unit
        args = (groups[pre], groups[pair.split("_to_")[1]], f"{prefix}_{pre}", edges, w)
        if "stp" in spec:
            synapses[pair] = stp_synapses(*args, _stp_values(design, spec["stp"], n, rng),
                                          _delays_ms(design, spec, n, rng), f"synapses_{pair}")
        else:
            synapses[pair] = static_synapses(*args, _delays_ms(design, spec, n, rng), f"synapses_{pair}")
    extra = []
    if "noise" in design:   # an independent Poisson train per neuron, into the X channel
        for pop in ("E", "I"):
            source = b2.PoissonGroup(sizes[pop], rates=design["noise"]["rate_hz"] * b2.Hz,
                                     name=f"noise_{pop.lower()}")
            w = draw_weights(design["noise"]["weight"], sizes[pop], rng) * unit
            index = np.arange(sizes[pop])
            extra += [source, static_synapses(source, groups[pop], f"{prefix}_X", (index, index), w,
                                              np.zeros(sizes[pop]), f"synapses_noise_{pop}")]

    spikes = {pop: b2.SpikeMonitor(groups[pop], name=f"spikes_{pop.lower()}") for pop in ("E", "I", "X")}
    network = b2.Network(*groups.values(), *synapses.values(), *extra, *spikes.values())
    return Block3Network(network=network, group_e=groups["E"], group_i=groups["I"], group_x=groups["X"],
                         synapses=synapses, spikes_e=spikes["E"], spikes_i=spikes["I"], spikes_x=spikes["X"])
