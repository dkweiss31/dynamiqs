from __future__ import annotations

from jaxtyping import PyTree

from ...result import Saved, SEResult
from ...time_array import Shape
from ...utils.operators import eye
from .._utils import _cartesian_vectorize, _flat_vectorize
from ..apis.sesolve import _sesolve
from ..core.abstract_integrator import SEPropagatorIntegrator


class SEPropagatorDynamiqsIntegrator(SEPropagatorIntegrator):
    def run(self) -> PyTree:
        # Square matrix-matrix product is faster than batched matrix-vector
        # product, so rather than directly calling sesolve on batched input
        # states, we call _sesolve instead and specify the initial state as a
        # square matrix (the identity matrix).
        n_batch = (
            self.H.in_axes,
            Shape(self.y0.shape[:-2]),
            Shape(),
            Shape(),
            Shape(),
            Shape(),
            Shape()
        )
        out_axes = SEResult(False, False, False, False, 0, 0)
        f = _flat_vectorize(_sesolve, n_batch, out_axes)
        seresult = f(
            self.H,
            self.y0,
            self.ts,
            None,
            self.solver,
            self.gradient,
            self.options,
        )
        saved = Saved(seresult.states, None, None)
        return self.result(saved, seresult.infos)
