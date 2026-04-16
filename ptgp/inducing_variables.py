class InducingVariables:
    """Base class for inducing variables.

    Enables dispatch to different implementations for inter-domain,
    multiscale, or structured inducing points.
    """

    @property
    def num_inducing(self):
        raise NotImplementedError


class InducingPoints(InducingVariables):
    """Standard real-space inducing points.

    Parameters
    ----------
    Z : tensor or PyMC random variable, shape (M, D)
        Inducing point locations.
    """

    def __init__(self, Z):
        self.Z = Z

    @property
    def num_inducing(self):
        return self.Z.shape[0]
