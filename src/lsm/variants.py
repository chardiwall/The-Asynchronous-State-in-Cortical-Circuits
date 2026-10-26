"""The three LSM variants, each derived from block 3's paper network (SOM S-p.20-21).

faithful    keeps every ingredient the paper's argument needs -- dense Bernoulli(p) wiring,
            Dale's law, fast strong inhibition, shared X input -- and rescales the coupling
            to strong-coupling form, g_ab(N) = g_ab^paper * sqrt(N_b^paper / N_b), so the
            network stays in the balanced regime at reservoir size. At N = 5000 it IS block 3.
dense_weak  the same dense wiring at the paper's nS values: coupling too weak for balance.
maass       the classic liquid (Maass, Natschlaeger & Markram 2002): a 3-D column with
            connection probability C_ab * exp(-(D/lambda)^2), the paper's nS values.

Every variant shares the same sqrt-scaled X -> reservoir pathway, so the variants differ
only in recurrent wiring and coupling; the two non-faithful ones carry one calibrated
scalar, the input gain on g_EX/g_IX, that equalises their background E rate with faithful's.
"""
import copy
import math

import numpy as np

from block3.connectivity import generate_bernoulli_connectivity

PRE_POPULATION_SIZE_KEY = {"E": "n_excitatory", "I": "n_inhibitory", "X": "n_external"}


def network_params(config: dict, variant: str, n: int, n_external: int) -> dict:
    """block3.model.build_network's params for `variant` at N = n recurrent neurons."""
    lsm = config["lsm"]
    paper = config["spiking_network"]
    params = copy.deepcopy(paper)
    params["dt_ms"] = paper["simulation"]["dt_ms"]
    n_e = round(lsm["excitatory_fraction"] * n)
    params["populations"] = {"n_excitatory": n_e, "n_inhibitory": n - n_e, "n_external": n_external}

    g = params["conductances_nS"]
    recurrent = ("g_EE", "g_EI", "g_IE", "g_II")
    scaled = recurrent if lsm["variants"][variant]["coupling"] == "sqrt_scaled" else ()
    for name in (*scaled, "g_EX", "g_IX"):   # the X pathway is sqrt-scaled in every variant
        key = PRE_POPULATION_SIZE_KEY[name[-1]]
        g[name] *= math.sqrt(paper["populations"][key] / params["populations"][key])
    gain = lsm["input_gain"][variant]
    g["g_EX"] *= gain
    g["g_IX"] *= gain
    return params


def column_positions(config: dict, n_e: int, n_i: int, rng: np.random.Generator) -> dict:
    """Integer grid sites of a (length x 5 x 5) column, assigned to neurons at random."""
    a, b = config["lsm"]["column"]["cross_section"]
    length = math.ceil((n_e + n_i) / (a * b))
    sites = np.array([(x, y, z) for x in range(length) for y in range(a) for z in range(b)])
    chosen = sites[rng.permutation(len(sites))[: n_e + n_i]]
    return {"E": chosen[:n_e], "I": chosen[n_e:]}


def column_connect(config: dict, positions: dict, p_external: float):
    """A build_network `connect` rule: Maass's distance-dependent probability inside the
    reservoir, Bernoulli(p_external) from X. Pair names are post<-pre ("EI" = I -> E)."""
    column = config["lsm"]["column"]

    def connect(name, n_pre, n_post, rng, exclude_self):
        if name[1] == "X":
            return generate_bernoulli_connectivity(n_pre, n_post, p_external, rng, exclude_self)
        pre, post = positions[name[1]], positions[name[0]]
        distance_sq = ((pre[:, None, :] - post[None, :, :]) ** 2).sum(axis=2)
        prob = column["c"][name] * np.exp(-distance_sq / column["lambda"] ** 2)
        if exclude_self:
            np.fill_diagonal(prob, 0.0)
        pre_index, post_index = np.nonzero(rng.random(prob.shape) < prob)
        return pre_index.astype(np.int32), post_index.astype(np.int32)

    return connect


def build_connect(config: dict, variant: str, params: dict, rng: np.random.Generator):
    """The `connect` rule for `variant`, or None for block 3's own Bernoulli draw."""
    if config["lsm"]["variants"][variant]["topology"] != "column":
        return None
    sizes = params["populations"]
    positions = column_positions(config, sizes["n_excitatory"], sizes["n_inhibitory"], rng)
    return column_connect(config, positions, params["connection_probability"])
