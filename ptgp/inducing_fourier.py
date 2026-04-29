import numpy as np
import pytensor.tensor as pt

from ptgp.inducing import InducingVariables
from ptgp.kernels.base import Kernel

# Basis ordering for the Fourier features: BLOCK ordering, matching the
# st--/VFF reference (Hensman, Durrande, Solin 2017). With K = num_frequencies
# and omegas = 2*pi*[0..K]/(b-a), the basis is laid out as
#     phi_0..phi_K       = cos(omega_k * (x - a)) for k = 0..K
#     phi_{K+1}..phi_{2K} = sin(omega_k * (x - a)) for k = 1..K
# The omega_0 sine row is dropped (sin(0) = 0). Total M = 2K + 1.
# This ordering must be matched by Kuf (Task 13) and any downstream consumer.


class FourierFeatures1D(InducingVariables):
    """1D Variational Fourier Features for Matern-1/2, 3/2, 5/2.

    Parameters
    ----------
    a, b : float
        Domain endpoints with a < b.
    num_frequencies : int
        Number of Fourier frequencies. ``num_inducing = 2 * num_frequencies + 1``.
    allow_extrapolation : bool, default False
        Suppress the runtime domain check installed at compile time.
    """

    def __init__(self, a, b, num_frequencies, *, allow_extrapolation=False):
        if not (float(a) < float(b)):
            raise ValueError(f"FourierFeatures1D requires a < b, got a={a}, b={b}.")
        if (float(b) - float(a)) < 1e-12:
            raise ValueError(f"Domain width b-a={b - a} is below numerical floor.")
        if int(num_frequencies) < 1:
            raise ValueError(f"num_frequencies must be >= 1, got {num_frequencies}.")
        self.a = float(a)
        self.b = float(b)
        self.num_frequencies = int(num_frequencies)
        self.allow_extrapolation = bool(allow_extrapolation)

    @property
    def num_inducing(self):
        return 2 * self.num_frequencies + 1

    @classmethod
    def from_data(cls, X, num_frequencies, buffer=0.1, *, allow_extrapolation=False):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError(
                f"FourierFeatures1D.from_data requires X of shape (N, 1), got {X.shape}."
            )
        lo, hi = float(X.min()), float(X.max())
        width = hi - lo
        return cls(
            a=lo - buffer * width,
            b=hi + buffer * width,
            num_frequencies=num_frequencies,
            allow_extrapolation=allow_extrapolation,
        )

    def _resolve_scaled_matern(self, kernel):
        """Recursively unwrap (scalar*…)*Matern{12,32,52} into (scale_tensor, base)."""
        from ptgp.kernels.combination import ProductKernel
        from ptgp.kernels.stationary import Matern12, Matern32, Matern52

        supported = (Matern12, Matern32, Matern52)

        def walk(node):
            if isinstance(node, ProductKernel):
                s1, b1 = walk(node.k1)
                s2, b2 = walk(node.k2)
                if b1 is not None and b2 is not None:
                    raise NotImplementedError(
                        f"FourierFeatures1D cannot factor a product of two kernels "
                        f"({type(b1).__name__} * {type(b2).__name__})."
                    )
                return s1 * s2, (b1 if b1 is not None else b2)
            if isinstance(node, Kernel):
                if not isinstance(node, supported):
                    raise NotImplementedError(
                        f"FourierFeatures1D supports Matern12/32/52 only; "
                        f"got {type(node).__name__}."
                    )
                return pt.as_tensor(1.0), node
            return pt.as_tensor(node), None

        scale, base = walk(kernel)
        if base is None:
            raise NotImplementedError(
                f"FourierFeatures1D requires a supported Matern leaf; got {kernel!r}."
            )
        return scale, base

    def _structured_Kuu_base(self, base_kernel):
        """Closed-form ``(d, U)`` such that ``Kuu(base) = diag(d) + U @ U.T``.

        Operates on a **unit-variance** Matern{12,32,52} ``base_kernel``; any
        amplitude prefactor is expected to have been peeled off by
        :meth:`_resolve_scaled_matern`. Uses block ordering (cos block then sin
        block, omega_0 sine row dropped). See Hensman et al. 2017 App. A and
        the st--/VFF reference.

        Returns
        -------
        d : pytensor tensor of shape (M,)
        U : pytensor tensor of shape (M, R), with R = 1, 2, 3 for
            Matern12, Matern32, Matern52 respectively.
        """
        from ptgp.kernels.stationary import Matern12, Matern32, Matern52

        a, b = self.a, self.b
        K = self.num_frequencies
        M = 2 * K + 1
        omegas = 2.0 * np.pi * np.arange(K + 1) / (b - a)  # length K+1
        omegas_sin = omegas[1:]  # drop omega_0 = 0
        ls = base_kernel.ls
        width = b - a
        n_cos = K + 1
        n_sin = K

        ones_cos = pt.as_tensor(np.ones(n_cos))
        zeros_cos_1 = pt.as_tensor(np.zeros((n_cos, 1)))
        zeros_sin_1 = pt.as_tensor(np.zeros((n_sin, 1)))

        if isinstance(base_kernel, Matern12):
            lamb = 1.0 / ls
            two_or_four = np.where(omegas == 0, 2.0, 4.0)
            d_cos = width * (lamb**2 + omegas**2) / lamb / two_or_four
            d_sin = width * (lamb**2 + omegas_sin**2) / lamb / 4.0

            d = pt.concatenate([pt.as_tensor(d_cos), pt.as_tensor(d_sin)], axis=0)
            # U has rank 1: cos block all ones, sin block zeros.
            U_cos = pt.reshape(ones_cos, (n_cos, 1))
            U_sin = pt.as_tensor(np.zeros((n_sin, 1)))
            U = pt.concatenate([U_cos, U_sin], axis=0)
            return d, U

        if isinstance(base_kernel, Matern32):
            lamb = np.sqrt(3.0) / ls
            four_or_eight = np.where(omegas == 0, 4.0, 8.0)
            d_cos = width * (lamb**2 + omegas**2) ** 2 / lamb**3 / four_or_eight
            d_sin = width * (lamb**2 + omegas_sin**2) ** 2 / lamb**3 / 8.0
            v_sin = omegas_sin / lamb

            d = pt.concatenate([pt.as_tensor(d_cos), pt.as_tensor(d_sin)], axis=0)
            # U has rank 2: column 0 covers cos block, column 1 covers sin block.
            U_cos_col = pt.concatenate([pt.reshape(ones_cos, (n_cos, 1)), zeros_cos_1], axis=1)
            U_sin_col = pt.concatenate([zeros_sin_1, pt.reshape(v_sin, (n_sin, 1))], axis=1)
            U = pt.concatenate([U_cos_col, U_sin_col], axis=0)
            return d, U

        if isinstance(base_kernel, Matern52):
            lamb = np.sqrt(5.0) / ls
            sixteen_or_32 = np.where(omegas == 0, 16.0, 32.0)
            v1 = (3.0 * (omegas / lamb) ** 2 - 1.0) / np.sqrt(8.0)
            # W_cos shape (n_cos, 2): col 0 = v1, col 1 = ones
            W_cos = pt.concatenate(
                [pt.reshape(v1, (n_cos, 1)), pt.reshape(ones_cos, (n_cos, 1))],
                axis=1,
            )
            d_cos = 3.0 * width / sixteen_or_32 / lamb**5 * (lamb**2 + omegas**2) ** 3

            v_sin = np.sqrt(3.0) * omegas_sin / lamb
            d_sin = 3.0 * width / 32.0 / lamb**5 * (lamb**2 + omegas_sin**2) ** 3

            d = pt.concatenate([pt.as_tensor(d_cos), pt.as_tensor(d_sin)], axis=0)
            # U has rank 3: cos block fills first 2 columns, sin block fills col 2.
            U_cos = pt.concatenate([W_cos, zeros_cos_1], axis=1)
            U_sin = pt.concatenate(
                [pt.as_tensor(np.zeros((n_sin, 2))), pt.reshape(v_sin, (n_sin, 1))],
                axis=1,
            )
            U = pt.concatenate([U_cos, U_sin], axis=0)
            return d, U

        raise NotImplementedError(
            f"_structured_Kuu_base supports Matern12/32/52 only; got {type(base_kernel).__name__}."
        )
