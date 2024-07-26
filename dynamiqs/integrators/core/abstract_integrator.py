from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import PyTree, Scalar

from ..._utils import _concatenate_sort
from ...gradient import Gradient
from ...options import Options
from ...result import MEResult, Result, Saved, SEPropagatorResult, SEResult, MEPropagatorResult
from ...solver import Solver
from ...time_array import TimeArray
from ...utils.utils import expect


class AbstractIntegrator(eqx.Module):
    @abstractmethod
    def run(self) -> PyTree:
        pass


class BaseIntegrator(AbstractIntegrator):
    ts: Array
    y0: Array
    H: TimeArray
    Es: Array
    solver: Solver
    gradient: Gradient | None
    options: Options

    @property
    def t0(self) -> Scalar:
        return self.ts[0] if self.options.t0 is None else self.options.t0

    @property
    def t1(self) -> Scalar:
        return self.ts[-1]

    @property
    def discontinuity_ts(self) -> Array | None:
        return self.H.discontinuity_ts

    def save(self, y: PyTree) -> Saved:
        ysave, Esave, extra = None, None, None
        if self.options.save_states:
            ysave = y
        if self.Es is not None and len(self.Es) > 0:
            Esave = expect(self.Es, y)
        if self.options.save_extra is not None:
            extra = self.options.save_extra(y)

        return Saved(ysave, Esave, extra)

    def collect_saved(self, saved: Saved, ylast: Array) -> Saved:
        # if save_states is False save only last state
        if not self.options.save_states:
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast, is_leaf=lambda x: x is None
            )

        # reorder Esave after jax.lax.scan stacking (ntsave, nE) -> (nE, ntsave)
        Esave = saved.Esave
        if Esave is not None:
            Esave = Esave.swapaxes(-1, -2)
            saved = eqx.tree_at(lambda x: x.Esave, saved, Esave)

        return saved

    @abstractmethod
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        pass


class MEIntegrator(BaseIntegrator):
    Ls: list[TimeArray]

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [x.discontinuity_ts for x in [self.H, *self.Ls]]
        return _concatenate_sort(*ts)


class SESolveIntegrator(BaseIntegrator):
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return SEResult(self.ts, self.solver, self.gradient, self.options, saved, infos)


class MESolveIntegrator(MEIntegrator):
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return MEResult(self.ts, self.solver, self.gradient, self.options, saved, infos)


class SEPropagatorIntegrator(BaseIntegrator):
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return SEPropagatorResult(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )


class MEPropagatorIntegrator(MEIntegrator):
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return MEPropagatorResult(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )
