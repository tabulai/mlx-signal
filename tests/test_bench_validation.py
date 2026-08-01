import numpy as np
import pytest
from bench._validation import rel_err, resample_quality


def test_rel_err_rejects_shape_error_and_nonfinite_values():
    ref = np.ones((2, 20), dtype=np.float32)
    with pytest.raises(RuntimeError, match="shape"):
        rel_err(ref[:, :-1], ref)
    with pytest.raises(RuntimeError, match="non-finite"):
        rel_err(np.full_like(ref, np.nan), ref)


def test_resample_quality_accepts_small_filter_difference_and_global_lag(rng):
    ref = rng.standard_normal((4, 4096)).astype(np.float32)
    out = np.pad(ref[:, :-2], ((0, 0), (2, 0)))
    out = 0.99 * out + 0.01 * rng.standard_normal(out.shape).astype(np.float32)
    result = resample_quality(out, ref, max_lag=8)
    assert result.startswith("r=")


def test_resample_quality_allows_one_sample_rate_rounding(rng):
    ref = rng.standard_normal((4, 4096)).astype(np.float32)
    assert resample_quality(ref[..., :-1], ref, max_lag=8).startswith("r=")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda x: -x, "signed correlation"),
        (lambda x: 1.18 * x, "NRMSE"),
        (lambda x: 1.06 * x, "gain error"),
        (lambda x: x + 0.01 * np.sqrt(np.mean(x * x)), "normalized bias"),
        (lambda x: x[..., :-10], "output length"),
        (lambda x: np.pad(x, ((0, 0), (0, 10))), "output length"),
    ],
    ids=["polarity", "nrmse", "gain", "bias", "truncated", "extra-tail"],
)
def test_resample_quality_rejects_wrong_outputs(rng, mutate, match):
    ref = rng.standard_normal((4, 4096)).astype(np.float32)
    with pytest.raises(RuntimeError, match=match):
        resample_quality(mutate(ref), ref, max_lag=8)


def test_resample_quality_checks_every_channel(rng):
    ref = rng.standard_normal((4, 4096)).astype(np.float32)
    out = ref.copy()
    out[-1] *= -1
    with pytest.raises(RuntimeError, match="signed correlation"):
        resample_quality(out, ref, max_lag=8)
