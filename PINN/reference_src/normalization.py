"""Feature normalizer for PINNModel inputs.

:class:`FeatureNormalizer` standardizes each input feature to zero mean and
unit variance. Normalization statistics are computed once on a training set
via :meth:`fit` and then applied to any split via :meth:`transform`.

The normalizer is implemented as a plain Python class (not a
:class:`torch.nn.Module`) so that it can be serialized together with the
checkpoint via ``torch.save`` / ``torch.load``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["FeatureNormalizer"]


class FeatureNormalizer:
    """Standardize features to zero mean, unit variance.

    After fitting on a training matrix :math:`X \\in \\mathbb{R}^{N \\times F}`,
    the transform is:

    .. math::

        X'_{:,j} = \\frac{X_{:,j} - \\mu_j}{\\sigma_j}

    where :math:`\\mu_j = \\mathrm{mean}(X_{:,j})` and
    :math:`\\sigma_j = \\mathrm{std}(X_{:,j})`.  Features with zero standard
    deviation (constant columns) are left unchanged (effective
    :math:`\\sigma_j = 1`).

    Attributes:
        mean_: Per-feature means, shape ``(F,)``.  ``None`` before :meth:`fit`.
        std_: Per-feature standard deviations (≥ 1e-8), shape ``(F,)``.
            ``None`` before :meth:`fit`.
        n_features_: Number of input features seen during fit.  ``None`` before
            :meth:`fit`.
    """

    def __init__(self) -> None:
        self.mean_: Tensor | None = None
        self.std_: Tensor | None = None
        self.n_features_: int | None = None

    # ------------------------------------------------------------------
    # Fit / transform API.
    # ------------------------------------------------------------------

    def fit(self, X: Tensor) -> "FeatureNormalizer":
        """Compute normalization statistics from *X*.

        Args:
            X: Training data, shape ``(N, F)``.  Must be a 2-D float tensor.

        Returns:
            ``self`` (for chaining with :meth:`transform`).

        Raises:
            ValueError: If *X* is not 2-D.
        """
        if X.dim() != 2:
            raise ValueError(
                f"FeatureNormalizer.fit expects a 2-D tensor, got {X.dim()}-D"
            )
        self.mean_ = X.mean(dim=0)                         # (F,)
        std = X.std(dim=0, unbiased=False)                 # (F,)
        # Replace near-zero std with 1 to avoid division by zero.
        self.std_ = std.where(std >= 1e-8, torch.ones_like(std))
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: Tensor) -> Tensor:
        """Apply the fitted normalization to *X*.

        Args:
            X: Input tensor, shape ``(N, F)`` or ``(F,)``.

        Returns:
            Standardized tensor of the same shape as *X*.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError(
                "FeatureNormalizer.transform called before fit()"
            )
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: Tensor) -> Tensor:
        """Fit on *X* and return its standardized form.

        Equivalent to calling :meth:`fit` followed by :meth:`transform`.

        Args:
            X: Training data, shape ``(N, F)``.

        Returns:
            Standardized version of *X*, shape ``(N, F)``.
        """
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Serialization helpers.
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Tensor | None]:
        """Return a dict that can be saved via ``torch.save``.

        Returns:
            Dict with keys ``'mean'``, ``'std'``, ``'n_features'``.
        """
        return {
            "mean": self.mean_,
            "std": self.std_,
            "n_features": (
                torch.tensor(self.n_features_)
                if self.n_features_ is not None
                else None
            ),
        }

    def load_state_dict(
        self, state: dict[str, Tensor | None]
    ) -> "FeatureNormalizer":
        """Restore statistics from a dict returned by :meth:`state_dict`.

        Args:
            state: Dict with keys ``'mean'``, ``'std'``, ``'n_features'``.

        Returns:
            ``self``.
        """
        self.mean_ = state.get("mean")
        self.std_ = state.get("std")
        n = state.get("n_features")
        self.n_features_ = int(n.item()) if n is not None else None
        return self
