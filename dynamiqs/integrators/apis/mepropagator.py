from __future__ import annotations

from functools import partial
import logging

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...options import Options
from ...result import SEPropagatorResult
from ...solver import Dopri5, Dopri8, Euler, Expm, Kvaerno3, Kvaerno5, Solver, Tsit5
from ...time_array import TimeArray
from ...utils.operators import eye
from .._utils import _astimearray, catch_xla_runtime_error, get_integrator_class, ispwc
from ..mepropagator.dynamiqs_integrator import MEPropagatorDynamiqsIntegrator
from ..mepropagator.expm_integrator import MEPropagatorExpmIntegrator


def mepropagator(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    tsave: ArrayLike,
    *,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> SEPropagatorResult:
    r"""Compute the propagator of the Lindblad master equation.
    """  # noqa: E501
    # === convert arguments
    H = _astimearray(H)
    jump_ops = [_astimearray(L) for L in jump_ops]
    tsave = jnp.asarray(tsave)

    # === check arguments
    _check_mepropagator_args(H, jump_ops)
    tsave = check_times(tsave, 'tsave')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _mepropagator(H, jump_ops, tsave, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _mepropagator(
    H: TimeArray,
    jump_ops: list[TimeArray],
    tsave: Array,
    solver: Solver | None,
    gradient: Gradient | None,
    options: Options,
) -> SEPropagatorResult:
    if not ispwc(H):
        raise NotImplementedError("We do not yet support non pwc Hamiltonians")
    solver = Expm

    integrators = {
        Expm: MEPropagatorExpmIntegrator,
    }
    integrator_class = get_integrator_class(integrators, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === initialize identity matrix
    broadcast_shape = jnp.broadcast_shapes(
        H.shape[:-2], *[jump_op.shape[:-2] for jump_op in jump_ops]
    )
    rho0 = eye(H.shape[-1]**2)
    rho0 = jnp.broadcast_to(rho0, broadcast_shape + rho0.shape[-2:])

    # === init integrator
    integrator = integrator_class(tsave, rho0, H, None, solver, gradient, options, jump_ops)

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_mepropagator_args(H: TimeArray, jump_ops: list[TimeArray]):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check jump_ops shape
    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(jump_ops) == 0:
        logging.warning(
            'Argument `jump_ops` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )
