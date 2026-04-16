# register rewrites at import time
import ptgp.linalg.rewrites  # noqa: F401

from ptgp.linalg.operator import LinearOperatorType
from ptgp.linalg.ops import KernelLinearOp, LinearOpLogdet, LinearOpMatvec, LinearOpSolve

__all__ = [
    "LinearOperatorType",
    "KernelLinearOp",
    "LinearOpMatvec",
    "LinearOpSolve",
    "LinearOpLogdet",
]
