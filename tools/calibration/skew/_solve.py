"""Shared design-matrix least-squares core for skew extraction (#140 SP-5b)."""
import numpy as np


class RankDeficientError(ValueError):
    """Raised when the design matrix cannot identify the requested parameters."""


def solve_design_matrix(A, b, min_rank):
    """Least-squares solve of A @ x = b.

    A: (n_obs, n_params) array-like; b: (n_obs,) array-like.
    Returns (x, fit_residual) where fit_residual = ||A @ x - b||_2.
    Raises RankDeficientError if rank(A) < min_rank (geometry cannot identify).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != b.shape[0]:
        raise ValueError(f"shape mismatch: A={A.shape}, b={b.shape}")
    if np.linalg.matrix_rank(A) < min_rank:
        raise RankDeficientError(f"design-matrix rank {np.linalg.matrix_rank(A)} < required {min_rank}")
    x, _res, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    fit_residual = float(np.linalg.norm(A @ x - b))
    return x, fit_residual
