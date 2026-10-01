"""Shared config loader used by every block: no hardcoded constants."""
import pytest

from config import load_config, output_path


def test_load_config_reads_real_config_yaml():
    config = load_config("config.yaml")

    assert config["seed"] == 1234
    assert config["pair_model"]["neuron"]["tau_m_ms"] == 10.0
    assert config["pair_model"]["synapse"]["epsp_peak_mV"] == 0.75


def test_load_config_rejects_empty_file(tmp_path):
    # yaml.safe_load returns None for an empty file, breaking the "-> dict" contract;
    # a bare None causes a confusing TypeError far from the real cause (empty config.yaml
    # from a bad merge/partial write) wherever the caller does config["some_key"].
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")

    with pytest.raises(ValueError):
        load_config(str(empty_file))


def test_output_path_resolves_against_the_artifacts_root():
    config = load_config("config.yaml")

    assert output_path(config, "block1_sweep_csv") == "artifacts/block1_full_pass.csv"


# Every output path as the code hardcoded it before paths moved into config.yaml. This is
# the behaviour-preservation contract: the shipped config must resolve to exactly these,
# or previously-written artifacts orphan and the filenames the block READMEs document break.
PATHS_BEFORE_CONFIG = {
    "block1_sweep_csv": "artifacts/block1_full_pass.csv",
    "block1_fig1b": "artifacts/fig1b.png",
    "block1_fig1c": "artifacts/fig1c.png",
    "block1_fig1e": "artifacts/fig1e.png",
    "block1_fig1f": "artifacts/fig1f.png",
    "block2_explore_csv": "artifacts/block2_exploratory_pass.csv",
    "block2_sweep_dir": "artifacts/block2_full_pass",
    "block2_sweep_csv": "artifacts/block2_full_pass.csv",
    "block2_current_dir": "artifacts/block2_full_pass_current",
    "block2_current_csv": "artifacts/block2_full_pass_current.csv",
    "block2_panels_dir": "artifacts/block2_illustrative",
    "block2_fig2b": "artifacts/block2_illustrative/fig2b.png",
    "block2_fig2c": "artifacts/block2_illustrative/fig2c.png",
    "block2_fig2d": "artifacts/block2_illustrative/fig2d.png",
    "block2_fig2e": "artifacts/block2_illustrative/fig2e.png",
    "block2_fig2g": "artifacts/block2_illustrative/fig2g.png",
    "block3_networks_dir": "artifacts/block3_full_pass",
    "block3_networks_csv": "artifacts/block3_full_pass.csv",
    "block3_panels_dir": "artifacts/block3_panels",
    "block3_fig3a_raster_csv": "artifacts/block3_panels/fig3a_raster.csv",
    "block3_fig3a_tracking_csv": "artifacts/block3_panels/fig3a_tracking.csv",
    "block3_fig3b_correlations_csv": "artifacts/block3_panels/fig3b_correlations.csv",
    "block3_vm_ccg_dir": "artifacts/block3_vm_ccg",
    "block3_vm_ccg_csv": "artifacts/block3_vm_ccg.csv",
    "block3_fig3a": "artifacts/block3_panels/fig3a.png",
    "block3_fig3b": "artifacts/block3_panels/fig3b.png",
    "block3_fig3c": "artifacts/block3_panels/fig3c.png",
    "block3_fig3d": "artifacts/block3_panels/fig3d.png",
    "metrics_jsonl": "artifacts/metrics.jsonl",
    "run_log": "artifacts/run.log",
}


def test_shipped_config_resolves_every_output_to_its_pre_config_path():
    config = load_config("config.yaml")

    resolved = {key: output_path(config, key) for key in PATHS_BEFORE_CONFIG}

    assert resolved == PATHS_BEFORE_CONFIG


def test_outputs_table_has_no_keys_beyond_the_documented_set():
    # A key nothing resolves is dead config -- it would drift silently out of step.
    config = load_config("config.yaml")

    assert set(config["outputs"]) == set(PATHS_BEFORE_CONFIG)


def test_changing_the_artifacts_root_redirects_every_output():
    config = load_config("config.yaml")
    config["paths"]["artifacts"] = "/scratch/run42"

    resolved = [output_path(config, key) for key in config["outputs"]]

    assert all(p.startswith("/scratch/run42/") for p in resolved)
    assert output_path(config, "block1_sweep_csv") == "/scratch/run42/block1_full_pass.csv"


def test_an_unknown_output_key_raises_rather_than_composing_a_wrong_path():
    # A typo must fail loudly: silently returning "artifacts/None" would write a run's
    # results somewhere nothing reads them back from.
    config = load_config("config.yaml")

    with pytest.raises(KeyError):
        output_path(config, "blcok1_sweep_csv")
