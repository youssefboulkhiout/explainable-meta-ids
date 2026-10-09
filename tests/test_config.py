"""The YAML reference configuration must reproduce the in-code CONFIG/DATASETS
exactly, so that `--config configs/default.yaml` cannot change any result."""
import copy
import os

import meta_ids as P
from conftest import ROOT


def test_default_yaml_matches_reference_config():
    datasets_before = copy.deepcopy(P.DATASETS)
    cfg = P.load_config(os.path.join(ROOT, "configs", "default.yaml"), P.CONFIG)
    assert cfg == P.CONFIG
    assert P.DATASETS == datasets_before


def test_no_config_returns_unchanged_copy():
    cfg = P.load_config(None, P.CONFIG)
    assert cfg == P.CONFIG and cfg is not P.CONFIG


def test_smoke_yaml_equals_quick_flag():
    cfg = P.load_config(os.path.join(ROOT, "configs", "smoke_test.yaml"), P.CONFIG)
    quick = copy.deepcopy(P.CONFIG)
    quick["max_rows"] = 30_000
    quick["meta"].update(episodes=60, eval_episodes=20, adapt_steps=30)
    quick["cnn"].update(epochs=8)
    quick["xai"].update(n_explain=15, kernel_nsamples=100, lime_num_samples=500,
                        stability_instances=3, stability_perturbs=2)
    assert cfg == quick
