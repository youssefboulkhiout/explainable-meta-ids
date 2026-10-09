"""Fast checks of the public API (no data needed)."""
import numpy as np
import torch

import meta_ids as P


def test_public_names_exist():
    for name in ["CONFIG", "DATASETS", "META_MODELS", "load_and_clean_dataset",
                 "split_and_standardize", "select_kbest_features",
                 "balance_training_classes", "train_lightgbm", "train_cnn",
                 "meta_train", "global_few_shot_adapt", "evaluate_episodic",
                 "run_open_set_detection", "explain_model",
                 "run_benchmark_on_dataset"]:
        assert hasattr(P, name), name


def test_revision_package_imports():
    from meta_ids.revision import cli  # noqa: F401
    from meta_ids.revision.e3_calibration import tempered_softmax, expected_calibration_error
    logp = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
    p = tempered_softmax(logp, 1.0)
    assert np.allclose(p.sum(1), 1.0)
    assert 0.0 <= expected_calibration_error(p, np.array([0, 2])) <= 1.0


def test_novelty_scores_shapes_and_range():
    proba = np.array([[0.9, 0.05, 0.05], [1 / 3, 1 / 3, 1 / 3]])
    s = P.compute_novelty_scores(proba, ["msp", "entropy", "margin"])
    assert set(s) == {"MSP", "Entropy", "Margin"}
    for v in s.values():
        assert v.shape == (2,) and v[1] >= v[0]          # uniform = more novel


def test_meta_learner_forward_shape():
    P.set_seed()
    params = P.init_mlp_params(10, 4, (16, 8))
    x = torch.randn(5, 10, device=P.DEVICE)
    assert P.mlp_logits(x, params).shape == (5, 4)
    assert P.mlp_embed(x, params).shape == (5, 8)
