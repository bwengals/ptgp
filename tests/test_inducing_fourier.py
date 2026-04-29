import numpy as np
import pytensor.tensor as pt
import pytest

from ptgp.inducing_fourier import FourierFeatures1D
from ptgp.kernels.stationary import ExpQuad, Matern12, Matern32, Matern52
from tests._fixtures.vff_kuu_oracle import (
    oracle_kuu_matern12,
    oracle_kuu_matern32,
    oracle_kuu_matern52,
)


def test_init_validates_a_lt_b():
    with pytest.raises(ValueError, match="a < b"):
        FourierFeatures1D(a=1.0, b=0.0, num_frequencies=8)


def test_init_validates_num_frequencies():
    with pytest.raises(ValueError, match="num_frequencies"):
        FourierFeatures1D(a=0.0, b=1.0, num_frequencies=0)


def test_num_inducing_is_2k_plus_1():
    f = FourierFeatures1D(a=0.0, b=1.0, num_frequencies=8)
    assert f.num_inducing == 17


def test_from_data_validates_shape():
    with pytest.raises(ValueError, match="shape"):
        FourierFeatures1D.from_data(np.zeros((10, 2)), num_frequencies=8)


def test_from_data_buffer():
    X = np.linspace(0, 1, 100)[:, None]
    f = FourierFeatures1D.from_data(X, num_frequencies=8, buffer=0.1)
    assert f.a < 0.0
    assert f.b > 1.0


def _scale_eval(s):
    return float(s.eval()) if hasattr(s, "eval") else float(s)


def test_resolve_bare_matern():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    s, base = f._resolve_scaled_matern(Matern32(input_dim=1, ls=1.0))
    assert isinstance(base, Matern32)
    assert _scale_eval(s) == 1.0


def test_resolve_canonical_eta_squared():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    eta = pt.as_tensor(2.0)
    s, base = f._resolve_scaled_matern(eta**2 * Matern32(input_dim=1, ls=1.0))
    assert isinstance(base, Matern32)
    np.testing.assert_allclose(_scale_eval(s), 4.0, atol=1e-12)


def test_resolve_nested_product_chain():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    eta = pt.as_tensor(2.0)
    s, base = f._resolve_scaled_matern((eta**2 * 1.5) * Matern32(input_dim=1, ls=1.0))
    assert isinstance(base, Matern32)
    np.testing.assert_allclose(_scale_eval(s), 6.0, atol=1e-12)


def test_resolve_rejects_non_matern():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    with pytest.raises(NotImplementedError, match="Matern"):
        f._resolve_scaled_matern(ExpQuad(input_dim=1, ls=1.0))


def test_resolve_rejects_two_kernel_product():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    k = Matern12(input_dim=1, ls=1.0) * Matern32(input_dim=1, ls=1.0)
    with pytest.raises(NotImplementedError, match="product of two"):
        f._resolve_scaled_matern(k)


def test_structured_kuu_base_shapes():
    for cls, expected_R in [(Matern12, 1), (Matern32, 2), (Matern52, 3)]:
        f = FourierFeatures1D(0, 1, num_frequencies=5)
        k = cls(input_dim=1, ls=0.5)
        d, U = f._structured_Kuu_base(k)
        d_v, U_v = d.eval(), U.eval()
        assert d_v.shape == (11,)
        assert U_v.shape == (11, expected_R)


def test_structured_kuu_base_matches_oracle_matern12():
    f = FourierFeatures1D(a=-0.5, b=1.5, num_frequencies=10)
    k = Matern12(input_dim=1, ls=0.3)
    d, U = [t.eval() for t in f._structured_Kuu_base(k)]
    Kuu_struct = np.diag(d) + U @ U.T
    Kuu_oracle = oracle_kuu_matern12(a=-0.5, b=1.5, ms=np.arange(11), ls=0.3)
    np.testing.assert_allclose(Kuu_struct, Kuu_oracle, atol=1e-10)


def test_structured_kuu_base_matches_oracle_matern32():
    f = FourierFeatures1D(a=-0.5, b=1.5, num_frequencies=10)
    k = Matern32(input_dim=1, ls=0.3)
    d, U = [t.eval() for t in f._structured_Kuu_base(k)]
    Kuu_struct = np.diag(d) + U @ U.T
    Kuu_oracle = oracle_kuu_matern32(a=-0.5, b=1.5, ms=np.arange(11), ls=0.3)
    np.testing.assert_allclose(Kuu_struct, Kuu_oracle, atol=1e-10)


def test_structured_kuu_base_matches_oracle_matern52():
    f = FourierFeatures1D(a=-0.5, b=1.5, num_frequencies=10)
    k = Matern52(input_dim=1, ls=0.3)
    d, U = [t.eval() for t in f._structured_Kuu_base(k)]
    Kuu_struct = np.diag(d) + U @ U.T
    Kuu_oracle = oracle_kuu_matern52(a=-0.5, b=1.5, ms=np.arange(11), ls=0.3)
    np.testing.assert_allclose(Kuu_struct, Kuu_oracle, atol=1e-10)
