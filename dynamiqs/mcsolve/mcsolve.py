from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from equinox.internal import while_loop
from jax.random import PRNGKey
from jax import Array
from jaxtyping import ArrayLike

from .. import norm
from .._utils import cdtype
from ..core._utils import _astimearray, compute_vmap, get_solver_class
from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Dopri8, Euler, Solver, Tsit5
from ..time_array import TimeArray
from ..utils.utils import unit, dag, mpow
from .mcdiffrax import MCDopri5, MCDopri8, MCEuler, MCTsit5

__all__ = ['mcsolve']


def mcsolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    ntraj: int = 10,
    key: PRNGKey = PRNGKey(42),
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> Result:
    r"""Perform Monte-Carlo evolution, unraveling the master equation.

    We follow the algorithm outlined in Abdelhafez et al. to efficiently perform
    Monte-Carlo sampling. First the no-jump trajectory is computed for a state vector $\ket{\psi(t)}$ at time
    $t$, starting from an initial state $\ket{\psi_0}$, according to the Schrödinger
    equation with non-Hermitian Hamiltonian ($\hbar=1$)
    $$
        \frac{\dd\ket{\psi(t)}}{\dt} = -i (H(t)
            -i/2 \sum_{k=1}^{N}L_{k}^{\dag}(t)L_{k}(t) ) \ket{\psi(t)},
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$ and $\{L_k(t)\}$ is a
    collection of jump operators at time $t$. We then extract the norm of the state
    at the final time, and take this as a lower bound for random numbers sampled
    for other trajectories such that they all experience at least one jump.

    Quote: Time-dependent Hamiltonian or jump operators
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.constant()`](/python_api/time_array/constant.html),
        [`dq.pwc()`](/python_api/time_array/pwc.html),
        [`dq.modulated()`](/python_api/time_array/modulated.html), or
        [`dq.timecallable()`](/python_api/time_array/timecallable.html).

    Quote: Running multiple simulations concurrently
        The Hamiltonian `H`, the jump operators `jump_ops` and the
         initial state `psi0` can be batched to solve multiple monte-carlo equations concurrently.
        All other arguments are common to every batch.

    Args:
        H _(array-like or time-array of shape (bH?, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, of shape (nL, n, n))_: List of
            jump operators.
        psi0 _(array-like of shape (bpsi?, n, 1))_: Initial state.
        tsave _(array-like of shape (nt,))_: Times at which the states and expectation
            values are saved. The equation is solved from `tsave[0]` to `tsave[-1]`, or
            from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        ntraj _(int, optional)_: Total number of trajectories to simulate, including
            the no-jump trajectory. Defaults to 10.
        key _(PRNGKeyArray, optional)_: random key to use for monte-carlo sampling.
        exp_ops _(list of array-like, of shape (nE, n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Tsit5()`](/python_api/solver/Tsit5.html).
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`](/python_api/options/Options.html).

    Returns:
        [`dq.Result`](/python_api/result/Result.html) object holding the result of the
            Monte-Carlo integration. It has the following attributes:

            - **states** _(array of shape (bH?, brho?, ntraj, nt, n, n))_ -- Saved states.
            - **expects** _(array of shape (bH?, brho?, nE, nt), optional)_ -- Saved
                expectation values.
            - **extra** _(PyTree, optional)_ -- Extra data saved with `save_extra()` if
                specified in `options`.
            - **infos** _(PyTree, optional)_ -- Solver-dependent information on the
                resolution.
            - **tsave** _(array of shape (nt,))_ -- Times for which states and
                expectation values were saved.
            - **solver** _(Solver)_ -- Solver used.
            - **gradient** _(Gradient)_ -- Gradient used.
            - **options** _(Options)_ -- Options used.
    """
    # === convert arguments
    H = _astimearray(H)
    jump_ops = [_astimearray(L) for L in jump_ops]
    psi0 = jnp.asarray(psi0, dtype=cdtype())
    tsave = jnp.asarray(tsave)
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None
    return _vmap_mcsolve(H, jump_ops, psi0, tsave, ntraj, key, exp_ops, solver, gradient, options)


def _vmap_mcsolve(
    H: TimeArray,
    jump_ops: list[TimeArray],
    psi0: Array,
    tsave: Array,
    ntraj: int,
    key: PRNGKey,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> Result:
    # === vectorize function
    # we vectorize over H, jump_ops and psi0, all other arguments are not vectorized
    # below we will have another layer of vectorization over ntraj
    is_batched = (
        H.ndim > 2,
        [jump_op.ndim > 2 for jump_op in jump_ops],
        psi0.ndim > 2,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    # the result is vectorized over `saved`
    out_axes = Result(None, None, None, None, 0, 0)
    f = compute_vmap(_mcsolve, options.cartesian_batching, is_batched, out_axes)
    return f(H, jump_ops, psi0, tsave, ntraj, key, exp_ops, solver, gradient, options)


def _mcsolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    ntraj: int,
    key: PRNGKey,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> Result:
    # TODO split earlier so that not reusing key for different batch dimensions
    key_1, key_2, key_3 = jax.random.split(key, num=3)
    # simulate no-jump trajectory
    rand0 = jnp.zeros(shape=(1, 1))
    no_jump_res = _single_traj(H, jump_ops, psi0, tsave, rand0, exp_ops, solver, gradient, options)
    # we have previously vmapped over batch dimensions
    # also want to exclude random number (which should be zero here)
    no_jump_state = no_jump_res.states[-1, 0:-1]
    # extract the no-jump probability
    p_nojump = jnp.einsum("id,id->", jnp.conj(no_jump_state), no_jump_state)
    # these random numbers define the moment when a jump occurs
    random_numbers = jax.random.uniform(key_2, shape=(ntraj, 1, 1), minval=p_nojump)
    # these keys will be used to randomly sample a jump operator
    traj_keys = jax.random.split(key_3, num=ntraj)
    # run all single trajectories at once
    f = jax.vmap(_jump_trajs, in_axes=(None, None, None, None, 0, 0, None, None, None, None))
    psis = f(H, jump_ops, psi0, tsave, traj_keys, random_numbers, exp_ops, solver, gradient, options)
    return no_jump_state, psis, p_nojump


def _jump_trajs(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    key: PRNGKey,
    rand: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
):
    res_before_jump = _single_traj(H, jump_ops, psi0, tsave, rand, exp_ops, solver, gradient, options)
    t_jump = res_before_jump.final_time[0]
    psi_before_jump = res_before_jump.states[-1, 0:-1]
    jump_op = sample_jump_ops(t_jump, psi_before_jump, jump_ops, key)
    psi_after_jump = unit(jump_op @ psi_before_jump)
    #TODO again not right for exp_ops
    new_tsave = jnp.linspace(t_jump, tsave[-1], 2)
    # for now just including one jump
    final_result = _single_traj(H, jump_ops, psi_after_jump, new_tsave,
                                jnp.zeros(shape=(1, 1)), exp_ops, solver, gradient, options)
    return res_before_jump, final_result


@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _single_traj(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    rand: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
):
    solvers = {
        Euler: MCEuler,
        Dopri5: MCDopri5,
        Dopri8: MCDopri8,
        Tsit5: MCTsit5,
    }
    solver_class = get_solver_class(solvers, solver)
    solver.assert_supports_gradient(gradient)
    state = jnp.concatenate((psi0, rand))
    mcsolver = solver_class(tsave, state, H, exp_ops, solver, gradient, options, jump_ops)
    return mcsolver.run()


def sample_jump_ops(t, psi, jump_ops, key, eps=1e-15):
    Ls = jnp.stack([L(t) for L in jump_ops])
    Lsd = dag(Ls)
    # i, j, k: hilbert dim indices; e: jump ops; d: index of dimension 1
    probs = jnp.einsum("id,eij,ejk,kd->e",
                       jnp.conj(psi), Lsd, Ls, psi
                       )
    logits = jnp.log(jnp.real(probs / (jnp.sum(probs)+eps)))
    # randomly sample the index of a single jump operator
    sample_idx = jax.random.categorical(key, logits, shape=(1,))
    # extract that jump operator and squeeze size 1 dims
    return jnp.squeeze(jnp.take(Ls, sample_idx, axis=0), axis=0)
