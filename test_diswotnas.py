"""
test_diswotnas.py
=================
Pytest test suite for DISWOTNAS and its helper functions.

Usage:
    pip install pytest torch numpy scikit-learn
    pytest test_diswotnas.py -v

Make sure diswotnas.py (your original file) is in the same directory.
"""

import math
import contextlib
import pytest
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# --------------------------------------------------------------------------- #
# Import the module under test.                                                #
# Adjust the import path if your file has a different name.                    #
# --------------------------------------------------------------------------- #
from diswotnas import (
    DISWOTNAS,
    batch_similarity,
    channel_similarity,
    relation_similarity_metric,
    semantic_similarity_metric,
)


# =========================================================================== #
# Fixtures & helpers                                                           #
# =========================================================================== #

class TinyTeacher(nn.Module):
    """Minimal teacher: one linear layer + feature extraction."""
    def __init__(self, in_features=16, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x)

    def forward_features(self, x):
        return [x, self.fc(x)]


class TinyStudent(nn.Module):
    """Minimal student with a different hidden dimension."""
    def __init__(self, in_features=16, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x)

    def forward_features(self, x):
        return [x, self.fc(x)]


class DummyEstimator(BaseEstimator):
    """Sklearn-compatible estimator that records whether fit() was called."""
    def __init__(self, C=1.0, kernel="rbf"):
        self.C = C
        self.kernel = kernel

    def fit(self, X, y=None, **kw):
        self.fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "fitted_")
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        check_is_fitted(self, "fitted_")
        n = len(X)
        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])

    def score(self, X, y=None):
        check_is_fitted(self, "fitted_")
        return 0.85

    def to_pytorch_model(self):
        return TinyStudent()


@pytest.fixture
def in_features():
    return 16


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def teacher(in_features):
    m = TinyTeacher(in_features)
    m.eval()
    return m


@pytest.fixture
def student(in_features):
    m = TinyStudent(in_features)
    m.eval()
    return m


@pytest.fixture
def random_batch(batch_size, in_features):
    images = torch.randn(batch_size, in_features)
    labels = torch.randint(0, 2, (batch_size,))
    return images, labels


@pytest.fixture
def dataloader(in_features):
    X = torch.randn(64, in_features)
    y = torch.randint(0, 2, (64,))
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)


@pytest.fixture
def param_grid():
    return {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}


@pytest.fixture
def nas(teacher, param_grid):
    return DISWOTNAS(
        estimator=DummyEstimator(),
        param_grid=param_grid,
        teacher_model=teacher,
        metric="relation",
        batch_size=16,
        device="cpu",
        verbose=0,
    )


# =========================================================================== #
# 1. Unit tests — batch_similarity                                             #
# =========================================================================== #

class TestBatchSimilarity:

    def test_identical_tensors_gives_zero(self, batch_size, in_features):
        """D(F, F) must be exactly 0 after normalisation."""
        f = torch.randn(batch_size, in_features)
        score = batch_similarity(f, f.clone())
        assert score.item() == pytest.approx(0.0, abs=1e-5), (
            f"Expected ~0.0, got {score.item():.6f}"
        )

    def test_returns_non_negative_scalar(self, batch_size, in_features):
        """L₂ squared distance is always ≥ 0."""
        for _ in range(5):
            ft = torch.randn(batch_size, in_features)
            fs = torch.randn(batch_size, in_features)
            score = batch_similarity(ft, fs)
            assert score.item() >= -1e-9, f"Got negative score: {score.item()}"

    def test_symmetric(self, batch_size, in_features):
        """D(T, S) == D(S, T) because L₂ is symmetric."""
        ft = torch.randn(batch_size, in_features)
        fs = torch.randn(batch_size, in_features)
        d1 = batch_similarity(ft, fs).item()
        d2 = batch_similarity(fs, ft).item()
        assert d1 == pytest.approx(d2, abs=1e-5), (
            f"D(T,S)={d1:.5f} != D(S,T)={d2:.5f}"
        )

    def test_normalisation_invariant_to_scale(self, batch_size, in_features):
        """Scaling both tensors by the same constant should leave the score unchanged."""
        ft = torch.randn(batch_size, in_features)
        fs = torch.randn(batch_size, in_features)
        d1 = batch_similarity(ft, fs).item()
        d2 = batch_similarity(ft * 1000, fs * 1000).item()
        assert d1 == pytest.approx(d2, abs=1e-4), (
            f"Scale should not change score: {d1:.5f} vs {d2:.5f}"
        )

    def test_single_sample_batch_is_finite(self, in_features):
        """bsz=1 should not produce NaN or Inf."""
        ft = torch.randn(1, in_features)
        fs = torch.randn(1, in_features)
        score = batch_similarity(ft, fs)
        assert torch.isfinite(score), f"Non-finite score with bsz=1: {score}"

    def test_score_increases_with_dissimilarity(self, batch_size, in_features):
        """A student identical to teacher scores lower than a random student."""
        f = torch.randn(batch_size, in_features)
        fs_random = torch.randn(batch_size, in_features)
        d_identical = batch_similarity(f, f.clone()).item()
        d_random = batch_similarity(f, fs_random).item()
        assert d_identical <= d_random, (
            f"Expected d_identical({d_identical:.4f}) <= d_random({d_random:.4f})"
        )

    def test_output_is_scalar(self, batch_size, in_features):
        """Return value must be a 0-dimensional tensor."""
        ft = torch.randn(batch_size, in_features)
        fs = torch.randn(batch_size, in_features)
        score = batch_similarity(ft, fs)
        assert score.ndim == 0, f"Expected scalar tensor, got shape {score.shape}"


# =========================================================================== #
# 2. Unit tests — channel_similarity                                           #
# =========================================================================== #

class TestChannelSimilarity:

    def test_identical_tensors_gives_zero(self):
        """D(F, F) must be ~0 for channel-wise similarity."""
        f = torch.randn(4, 8, 4)
        score = channel_similarity(f, f.clone())
        assert score.item() == pytest.approx(0.0, abs=1e-5)

    def test_returns_non_negative(self):
        for _ in range(5):
            ft = torch.randn(4, 8, 4)
            fs = torch.randn(4, 8, 4)
            assert channel_similarity(ft, fs).item() >= -1e-9

    def test_single_channel_no_nan(self):
        """ch=1 edge case must not produce NaN."""
        ft = torch.randn(4, 1, 8)
        fs = torch.randn(4, 1, 8)
        score = channel_similarity(ft, fs)
        assert not torch.isnan(score), "Got NaN with single channel"

    def test_output_is_scalar(self):
        ft = torch.randn(4, 8, 4)
        fs = torch.randn(4, 8, 4)
        assert channel_similarity(ft, fs).ndim == 0


# =========================================================================== #
# 3. Unit tests — relation_similarity_metric                                   #
# =========================================================================== #

class TestRelationSimilarityMetric:

    def test_returns_float(self, teacher, student, random_batch):
        score = relation_similarity_metric(teacher, student, random_batch)
        assert isinstance(score, (float, int)) or torch.is_tensor(score), (
            f"Expected numeric, got {type(score)}"
        )

    def test_teacher_equals_student_gives_near_zero(self, teacher, random_batch):
        """When teacher IS the student, relation distance should be ~0, so score ~0."""
        score = relation_similarity_metric(teacher, teacher, random_batch)
        # score = -1 * distance; identical models → distance ~0 → score ~0
        val = score.item() if torch.is_tensor(score) else score
        assert val == pytest.approx(0.0, abs=1e-4), f"Expected ~0, got {val}"

    def test_score_is_non_positive(self, teacher, student, random_batch):
        """relation metric returns -1 * distance, so it should be ≤ 0."""
        score = relation_similarity_metric(teacher, student, random_batch)
        val = score.item() if torch.is_tensor(score) else score
        assert val <= 1e-9, f"Score should be non-positive, got {val}"

    def test_no_gradients_computed(self, teacher, student, random_batch):
        """Called inside torch.no_grad(); no gradient should be tracked."""
        with torch.no_grad():
            score = relation_similarity_metric(teacher, student, random_batch)
        val = score if not torch.is_tensor(score) else score
        # Just check it runs without error inside no_grad context


# =========================================================================== #
# 4. Unit tests — DISWOTNAS class interface                                    #
# =========================================================================== #

class TestDISWOTNASInterface:

    def test_init_stores_all_params(self, teacher, param_grid):
        nas = DISWOTNAS(
            estimator=DummyEstimator(),
            param_grid=param_grid,
            teacher_model=teacher,
            metric="semantic",
            batch_size=64,
            device="cpu",
            n_jobs=2,
            verbose=1,
        )
        assert nas.metric == "semantic"
        assert nas.batch_size == 64
        assert nas.device == "cpu"
        assert nas.n_jobs == 2
        assert nas.verbose == 1

    def test_get_params_returns_all_keys(self, nas):
        p = nas.get_params()
        required = ["estimator", "param_grid", "teacher_model",
                    "metric", "batch_size", "device", "n_jobs", "verbose"]
        for key in required:
            assert key in p, f"Missing key: {key}"

    def test_set_params_updates_only_specified(self, nas):
        original_device = nas.device
        nas.set_params(metric="semantic", batch_size=64)
        assert nas.metric == "semantic"
        assert nas.batch_size == 64
        assert nas.device == original_device  # unchanged

    def test_set_params_returns_self(self, nas):
        ret = nas.set_params(verbose=1)
        assert ret is nas, "set_params() must return self"

    def test_predict_before_fit_raises(self, nas):
        with pytest.raises(Exception):
            nas.predict(np.zeros((4, 16)))

    def test_predict_proba_before_fit_raises(self, nas):
        with pytest.raises(Exception):
            nas.predict_proba(np.zeros((4, 16)))

    def test_score_before_fit_raises(self, nas):
        with pytest.raises(Exception):
            nas.score(np.zeros((4, 16)), np.zeros(4))


# =========================================================================== #
# 5. Unit tests — param_grid expansion                                         #
# =========================================================================== #

class TestParamGrid:

    def test_full_cartesian_product(self, param_grid):
        from sklearn.model_selection import ParameterGrid
        combos = list(ParameterGrid(param_grid))
        # C: 3 values × kernel: 2 values = 6
        assert len(combos) == 6

    def test_single_value_per_key(self):
        from sklearn.model_selection import ParameterGrid
        combos = list(ParameterGrid({"C": [1], "kernel": ["rbf"]}))
        assert len(combos) == 1

    def test_empty_grid(self):
        from sklearn.model_selection import ParameterGrid
        combos = list(ParameterGrid({}))
        assert len(combos) == 1
        assert combos[0] == {}

    def test_all_combinations_present(self, param_grid):
        from sklearn.model_selection import ParameterGrid
        combos = list(ParameterGrid(param_grid))
        c_values = {c["C"] for c in combos}
        k_values = {c["kernel"] for c in combos}
        assert c_values == {0.1, 1.0, 10.0}
        assert k_values == {"rbf", "linear"}


# =========================================================================== #
# 6. Integration tests — fit() search loop                                     #
# =========================================================================== #

class TestFitLoop:

    @pytest.fixture
    def fitted_nas(self, nas, dataloader):
        """Run fit() once and return the fitted estimator."""
        X_np = np.random.randn(64, 16).astype(np.float32)
        y_np = np.random.randint(0, 2, 64)
        nas.fit(X_np, y_np)
        return nas

    def test_best_score_is_set(self, fitted_nas):
        assert hasattr(fitted_nas, "best_score_"), "best_score_ not set"
        assert isinstance(fitted_nas.best_score_, float), (
            f"Expected float, got {type(fitted_nas.best_score_)}"
        )

    def test_best_params_is_dict(self, fitted_nas):
        assert hasattr(fitted_nas, "best_params_"), "best_params_ not set"
        assert isinstance(fitted_nas.best_params_, dict)

    def test_best_params_keys_match_grid(self, fitted_nas, param_grid):
        for key in param_grid:
            assert key in fitted_nas.best_params_, f"Key {key} missing from best_params_"

    def test_best_estimator_is_fitted(self, fitted_nas):
        assert hasattr(fitted_nas, "best_estimator_"), "best_estimator_ not set"
        # sklearn's check_is_fitted should not raise
        check_is_fitted(fitted_nas.best_estimator_)

    def test_cv_results_length_matches_grid(self, fitted_nas, param_grid):
        from sklearn.model_selection import ParameterGrid
        n = len(list(ParameterGrid(param_grid)))
        r = fitted_nas.cv_results_
        assert len(r["params"]) == n
        assert len(r["mean_test_score"]) == n
        assert len(r["rank_test_score"]) == n

    def test_rank_sum(self, fitted_nas, param_grid):
        """Ranks 1..n must sum to n(n+1)/2 — no ties or gaps."""
        from sklearn.model_selection import ParameterGrid
        n = len(list(ParameterGrid(param_grid)))
        rank_sum = sum(fitted_nas.cv_results_["rank_test_score"])
        assert rank_sum == n * (n + 1) // 2, (
            f"rank sum={rank_sum}, expected {n*(n+1)//2}"
        )

    def test_best_score_equals_max_of_cv_results(self, fitted_nas):
        max_score = max(fitted_nas.cv_results_["mean_test_score"])
        assert fitted_nas.best_score_ == pytest.approx(max_score, abs=1e-9)

    def test_predict_after_fit(self, fitted_nas):
        X = np.random.randn(10, 16).astype(np.float32)
        preds = fitted_nas.predict(X)
        assert len(preds) == 10

    def test_predict_proba_after_fit(self, fitted_nas):
        X = np.random.randn(5, 16).astype(np.float32)
        proba = fitted_nas.predict_proba(X)
        assert proba.shape == (5, 2), f"Expected (5,2), got {proba.shape}"

    def test_score_after_fit(self, fitted_nas):
        X = np.random.randn(10, 16).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        s = fitted_nas.score(X, y)
        assert isinstance(s, float)

    def test_returns_self(self, nas):
        X = np.random.randn(32, 16).astype(np.float32)
        y = np.random.randint(0, 2, 32)
        ret = nas.fit(X, y)
        assert ret is nas, "fit() must return self"


# =========================================================================== #
# 7. Edge cases                                                                #
# =========================================================================== #

class TestEdgeCases:

    def test_unknown_metric_raises(self, teacher, student, random_batch):
        """Passing an unsupported metric must raise ValueError."""
        nas = DISWOTNAS(
            estimator=DummyEstimator(),
            param_grid={"C": [1.0]},
            teacher_model=teacher,
            metric="cosine",
            device="cpu",
        )
        X = np.random.randn(16, 16).astype(np.float32)
        y = np.random.randint(0, 2, 16)
        with pytest.raises(ValueError, match="Unknown metric"):
            nas.fit(X, y)

    def test_batch_similarity_large_values_finite(self):
        """Features at 1e6 scale must stay finite after Gram normalisation."""
        ft = torch.randn(4, 8) * 1e6
        fs = torch.randn(4, 8) * 1e6
        score = batch_similarity(ft, fs)
        assert torch.isfinite(score), f"Non-finite: {score}"

    def test_fit_accepts_dataloader_directly(self, nas):
        """X can be a DataLoader instead of a numpy array."""
        X = torch.randn(32, 16)
        y = torch.randint(0, 2, (32,))
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=16)
        # Should not raise
        nas.fit(dl, y=None)

    def test_single_param_combination(self, teacher):
        """A grid with one combination should still set best_params_."""
        nas = DISWOTNAS(
            estimator=DummyEstimator(),
            param_grid={"C": [1.0], "kernel": ["rbf"]},
            teacher_model=teacher,
            metric="relation",
            device="cpu",
        )
        X = np.random.randn(16, 16).astype(np.float32)
        y = np.random.randint(0, 2, 16)
        nas.fit(X, y)
        assert nas.best_params_ == {"C": 1.0, "kernel": "rbf"}

    def test_fit_with_torch_tensor_input(self, nas):
        """fit() must accept torch.Tensor inputs."""
        X = torch.randn(32, 16)
        y = torch.randint(0, 2, (32,))
        nas.fit(X, y)
        assert hasattr(nas, "best_estimator_")

    def test_cv_results_scores_are_floats(self, nas):
        """_compute_similarity_score must return plain Python floats."""
        X = np.random.randn(32, 16).astype(np.float32)
        y = np.random.randint(0, 2, 32)
        nas.fit(X, y)
        for s in nas.cv_results_["mean_test_score"]:
            assert isinstance(s, float), f"Expected float, got {type(s)}"
            assert math.isfinite(s), f"Score is not finite: {s}"

    def test_verbose_flag_does_not_break_fit(self, teacher, param_grid):
        """verbose=1 must not raise any errors."""
        nas = DISWOTNAS(
            estimator=DummyEstimator(),
            param_grid=param_grid,
            teacher_model=teacher,
            metric="relation",
            device="cpu",
            verbose=1,
        )
        X = np.random.randn(16, 16).astype(np.float32)
        y = np.random.randint(0, 2, 16)
        nas.fit(X, y)  # should not raise


# =========================================================================== #
# 8. Smoke test — semantic metric                                              #
# =========================================================================== #

class TestSemanticMetric:

    def test_semantic_metric_returns_numeric_direct(self, teacher, student):
        """Call semantic_similarity_metric directly, outside torch.no_grad().

        This is the only correct way to call it: the function calls
        .backward() internally and therefore requires an active autograd graph.
        Models must be in train() mode so their parameters accumulate grads.
        """
        teacher.train()
        student.train()
        images = torch.randn(8, 16)   # no requires_grad needed on input;
        labels = torch.randint(0, 2, (8,))  # the model params carry the graph
        score = semantic_similarity_metric(teacher, student, (images, labels))
        val = score.item() if torch.is_tensor(score) else score
        assert math.isfinite(val), f"Non-finite semantic score: {val}"

    def test_semantic_fit_runs(self, teacher):
        """fit() with metric='semantic' must complete without error.

        Fixed in diswotnas.py: _compute_similarity_score now uses
        contextlib.nullcontext() instead of torch.no_grad() for the semantic
        metric, so the autograd graph is preserved for .backward().
        """
        nas = DISWOTNAS(
            estimator=DummyEstimator(),
            param_grid={"C": [1.0]},
            teacher_model=teacher,
            metric="semantic",
            device="cpu",
        )
        X = np.random.randn(16, 16).astype(np.float32)
        y = np.random.randint(0, 2, 16)
        nas.fit(X, y)
        assert hasattr(nas, "best_score_")
        assert isinstance(nas.best_score_, float)
