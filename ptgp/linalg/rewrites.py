# PyTensor rewrite rules for GP linear algebra.
# Registered at import time via @register_stabilize / @register_canonicalize.
# Rules recognise PSD-annotated kernel matrices and substitute efficient
# implementations (Cholesky solve, shared factorisations, etc.).
