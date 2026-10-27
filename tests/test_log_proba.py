"""LightGBM log-probabilities from raw scores agree with predict_proba, without underflow."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from nesy_tabfm.models.salt_lgbm import lgbm_log_proba


@pytest.mark.parametrize("n_classes", [2, 5])
def test_lgbm_log_proba_matches_predict_proba(n_classes):
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(600, 4)), columns=list("abcd"))
    y = (x["a"] * 2 + x["b"] + rng.normal(scale=0.3, size=600)).rank(method="first").to_numpy()
    y = np.minimum((y * n_classes / len(y)).astype(int), n_classes - 1)
    model = lgb.LGBMClassifier(n_estimators=30, verbose=-1, random_state=0).fit(x, y)

    log_p = lgbm_log_proba(model, x)
    assert log_p.shape == (600, n_classes)
    np.testing.assert_allclose(np.exp(log_p), model.predict_proba(x), rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(model.classes_[np.argmax(log_p, axis=1)], model.predict(x))
    np.testing.assert_allclose(np.exp(log_p).sum(axis=1), 1.0, rtol=1e-12)
    assert np.isfinite(log_p).all()
