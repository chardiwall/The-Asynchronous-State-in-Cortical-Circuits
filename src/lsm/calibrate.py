"""The non-faithful variants' one free parameter, the input gain, set by bisection on log
gain (rate rises monotonically with it) to match the faithful variant's rate:

  calibrate       background-only E rate at lsm.size  -> lsm.input_gain (asynchrony runs)
  calibrate_task  mean reservoir rate during training stimuli -> lsm.task_input_gain
                  (classification runs). Matched at background, the unbalanced liquids run
                  several times faster than faithful under stimulus (maass ~200 Hz), so the
                  readout comparison is made at matched DRIVEN rate instead.
"""
import math

from lsm.asynchrony import run_background


def _bisect(rate_at, target: float, bounds, iterations: int) -> float:
    low, high = (math.log(g) for g in bounds)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if rate_at(math.exp(mid)) < target else (low, mid)
    return math.exp(0.5 * (low + high))


def _with_gain(config: dict, key: str, variant: str, gain: float) -> dict:
    lsm = config["lsm"]
    return {**config, "lsm": {**lsm, key: {**lsm[key], variant: gain}}}


def calibrate(config: dict) -> dict:
    lsm, settings = config["lsm"], config["lsm"]["calibration"]

    def e_rate(variant, gain):
        trial = _with_gain(config, "input_gain", variant, gain)
        rate = run_background(trial, variant, lsm["size"], 0, settings["length_s"],
                              record_currents=False)["rate_excitatory_hz"]
        print(f"  {variant} gain={gain:.3f} -> E {rate:.3f} Hz", flush=True)
        return rate

    target = e_rate("faithful", 1.0)
    gains = {"target_rate_excitatory_hz": target}
    for variant in lsm["variants"]:
        if variant != "faithful":
            gains[variant] = _bisect(lambda g: e_rate(variant, g), target,
                                     settings["gain_bounds"], settings["iterations"])
    return gains


def calibrate_task(config: dict, dataset: dict, train_samples: list) -> dict:
    from lsm.reservoir import chunk_features, thinning

    lsm, settings = config["lsm"], config["lsm"]["calibration"]
    q = thinning(config, dataset, train_samples)
    window_s = dataset["window_ms"] / 1000.0

    def rate(variant, gain):
        trial = _with_gain(config, "task_input_gain", variant, gain)
        x, _ = chunk_features(trial, dataset, variant, train_samples, q, 0)
        hz = float(x.sum() / (x.shape[0] * lsm["size"] * window_s))
        print(f"  {variant} gain={gain:.3f} -> {hz:.2f} Hz", flush=True)
        return hz

    target = rate("faithful", 1.0)
    gains = {"target_rate_hz": target}
    for variant in lsm["variants"]:
        if variant != "faithful":
            gains[variant] = _bisect(lambda g: rate(variant, g), target,
                                     settings["task_gain_bounds"], settings["iterations"])
    return gains
