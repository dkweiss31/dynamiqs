from __future__ import annotations

import warnings
from abc import abstractmethod

import diffrax as dx
import equinox as eqx
from jax import Array
from jaxtyping import PyTree

from ...gradient import Autograd, CheckpointAutograd
from .abstract_integrator import BaseIntegrator
from diffrax._progress_meter import NoProgressMeter


class DiffraxIntegrator(BaseIntegrator):
    # Subclasses should implement:
    # - the attributes: stepsize_controller, dt0, max_steps, diffrax_solver, terms
    # - the methods: result, infos

    stepsize_controller: dx.AbstractVar[dx.AbstractStepSizeController]
    dt0: dx.AbstractVar[float | None]
    max_steps: dx.AbstractVar[int]
    diffrax_solver: dx.AbstractVar[dx.AbstractSolver]
    terms: dx.AbstractVar[dx.AbstractTerm]
    event: dx.Event | None

    def __init__(self, *args):
        # pass all init arguments to `BaseIntegrator`
        super().__init__(*args)

    def run(self) -> PyTree:
        with warnings.catch_warnings():
            # TODO: remove once complex support is stabilized in diffrax
            warnings.simplefilter('ignore', UserWarning)
            # TODO: remove once https://github.com/patrick-kidger/diffrax/issues/445 is
            # closed
            warnings.simplefilter('ignore', FutureWarning)

            # === prepare diffrax arguments
            fn = lambda t, y, args: self.save(y)  # noqa: ARG005
            subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)  # save solution regularly
            subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
            saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

            if self.gradient is None:
                adjoint = dx.RecursiveCheckpointAdjoint()
            elif isinstance(self.gradient, CheckpointAutograd):
                adjoint = dx.RecursiveCheckpointAdjoint(self.gradient.ncheckpoints)
            elif isinstance(self.gradient, Autograd):
                adjoint = dx.DirectAdjoint()

            if self.event is None:
                progress_meter = self.options.progress_meter.to_diffrax()
            else:
                progress_meter = NoProgressMeter()

            # === solve differential equation with diffrax
            solution = dx.diffeqsolve(
                self.terms,
                self.diffrax_solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=self.y0,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                event=self.event,
                max_steps=self.max_steps,
                progress_meter=progress_meter,
            )

        # === collect and return results
        save_a, save_b = solution.ys
        saved = self.collect_saved(save_a, save_b[0])
        return self.result(saved, solution.ts[-1][0], infos=self.infos(solution.stats))

    @abstractmethod
    def infos(self, stats: dict[str, Array]) -> PyTree:
        pass


class FixedStepIntegrator(DiffraxIntegrator):
    # Subclasses should implement:
    # - the attributes: diffrax_solver, terms
    # - the methods: result

    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: fixed step solvers always make the same number of steps
                return (
                    f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    stepsize_controller: dx.AbstractStepSizeController = dx.ConstantStepSize()
    max_steps: int = 100_000  # TODO: fix hard-coded max_steps

    @property
    def dt0(self) -> float:
        return self.solver.dt

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(stats['num_steps'])


class EulerIntegrator(FixedStepIntegrator):
    diffrax_solver: dx.AbstractSolver = dx.Euler()


class AdaptiveStepIntegrator(DiffraxIntegrator):
    # Subclasses should implement:
    # - the attributes: diffrax_solver, terms
    # - the methods: result

    class Infos(eqx.Module):
        nsteps: Array
        naccepted: Array
        nrejected: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                return (
                    f'avg. {self.nsteps.mean():.1f} steps ({self.naccepted.mean():.1f}'
                    f' accepted, {self.nrejected.mean():.1f} rejected) | infos shape'
                    f' {self.nsteps.shape}'
                )
            return (
                f'{self.nsteps} steps ({self.naccepted} accepted,'
                f' {self.nrejected} rejected)'
            )

    dt0 = None

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        return dx.PIDController(
            rtol=self.solver.rtol,
            atol=self.solver.atol,
            safety=self.solver.safety_factor,
            factormin=self.solver.min_factor,
            factormax=self.solver.max_factor,
            jump_ts=self.discontinuity_ts,
        )

    @property
    def max_steps(self) -> int:
        return self.solver.max_steps

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(
            stats['num_steps'], stats['num_accepted_steps'], stats['num_rejected_steps']
        )


class Dopri5Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Dopri5()


class Dopri8Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Dopri8()


class Tsit5Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Tsit5()


class Kvaerno3Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Kvaerno3()


class Kvaerno5Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Kvaerno5()
