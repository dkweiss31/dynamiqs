from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from dynamiqs._utils import _concatenate_sort
from dynamiqs.result import Saved

from ...utils.utils.general import expm
from ...utils.vectorization import slindbladian
from .._utils import ispwc
from .abstract_integrator import BaseIntegrator


class ExpmIntegrator(BaseIntegrator):
    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: propagator solvers can make different number of steps between
                # batch elements when batching over PWC objects
                return (
                    f'avg. {self.nsteps.mean():.1f} steps | infos shape'
                    f' {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    def __init__(self, *args):
        super().__init__(*args)

        # check that Hamiltonian is time-independent
        if not ispwc(self.H):
            raise TypeError(
                'Solver `Expm` requires a piece-wise constant Hamiltonian.'
            )

    def _diff_eq_rhs(self, t: float) -> Array:
        raise NotImplementedError

    def collect_saved(self, saved: Saved, ylast: Array, times: Array) -> Saved:
        # === extract the propagators or states at the correct times
        # the -1 is because the indices are defined by t_diffs,
        # not times itself
        t_idxs = jnp.argmin(jnp.abs(times - self.ts[:, None]), axis=1)
        t_idxs = jnp.where(t_idxs > 0, t_idxs - 1, t_idxs)
        if self.options.save_states:
            states = saved.ysave[t_idxs]
            states = jnp.moveaxis(states, 0, -3)
            saved = eqx.tree_at(lambda x: x.ysave, saved, states)
        if saved.Esave is not None:
            saved = eqx.tree_at(lambda x: x.Esave, saved, saved.Esave[t_idxs])
        if saved.extra is not None:
            saved = eqx.tree_at(lambda x: x.extra, saved, saved.extra[t_idxs])
        return super().collect_saved(saved, ylast)

    def run(self) -> PyTree:
        # === find times at which the Hamiltonian changes (between t0 and tfinal)
        t0 = jnp.asarray(self.t0).reshape(-1)
        times = _concatenate_sort(t0, self.discontinuity_ts, self.ts)
        inbounds = (times[:-1] >= t0) & (times[1:] <= self.ts[-1])
        t_diffs = jnp.where(inbounds, jnp.diff(times), 0.0)

        # === compute the propagators
        # don't need the last time in times since the hamiltonian is guaranteed
        # to be constant over the region times[-2] to times[-1]
        Hs = jnp.stack([self._diff_eq_rhs(t) for t in times[:-1]])
        dims = tuple(range(-Hs.ndim + 1, 0))
        t_diffs = jnp.expand_dims(t_diffs, dims)
        step_propagators = expm(t_diffs * Hs)

        # === combine propagators to get the propagator at each requested time
        def _reduce(prev_prop: Array, next_prop: Array) -> Array:
            # notice the ordering of prev_prop and next_prop, want
            # next_prop to be to the left of prev_prop
            total_prop = next_prop @ prev_prop
            return total_prop, self.save(total_prop)

        # note that ylast will correspond to the correct final state/propagator
        # even if self.ts[-1] is not the final time in the scan, since the
        # t_diffs for these later times have been set to zero
        ylast, saved = jax.lax.scan(_reduce, self.y0, step_propagators)

        # === collect and return results
        nsteps = (t_diffs != 0).sum()
        saved = self.collect_saved(saved, ylast, times)
        return self.result(saved, infos=self.Infos(nsteps))


class SEExpmIntegrator(ExpmIntegrator):
    def _diff_eq_rhs(self, t: float) -> Array:
        return -1j * self.H(t)


class MEExpmIntegrator(ExpmIntegrator):
    def _diff_eq_rhs(self, t: float) -> Array:
        return slindbladian(self.H(t), [L(t) for L in self.Ls])
