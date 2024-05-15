from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import ArrayLike

import dynamiqs as dq
from dynamiqs import Options
from dynamiqs.utils.fidelity import infidelity_coherent, infidelity_incoherent
from dynamiqs.utils.file_io import append_to_h5, write_to_h5_multi

from .._utils import cdtype
from ..solver import Solver, Tsit5
from ..time_array import CallableTimeArray, timecallable, BatchedCallable

__all__ = ["grape"]


def grape(
    H_func: BatchedCallable,
    initial_states: ArrayLike,
    target_states: ArrayLike,
    tsave: ArrayLike,
    params_to_optimize: ArrayLike,
    *,
    grape_type="unitary",
    jump_ops=None,
    target_states_traj: ArrayLike = None,
    filepath: str = "tmp.h5py",
    optimizer: optax.GradientTransformation = optax.adam(0.1, b1=0.99, b2=0.99),
    solver: Solver = Tsit5(),
    options: Options = Options(),
    init_params_to_save: dict = {},
) -> ArrayLike:
    r"""Perform gradient descent to optimize Hamiltonian parameters

        This function takes as input a list of initial_states and a list of
        target_states, and optimizes params_to_optimize to achieve the highest fidelity
        state transfer. It saves the parameters from every epoch and the associated fidelity
        in the file filepath

        Args:
             H_func _(PyTree object)_: Hamiltonian. Assumption is that we can
                instantiate a timecallable instance with
                H_func = partial(H_func, drive_params=params_to_optimize)
                H = timecallable(H_func, )
             initial_states _(list of array-like of shape (n, 1))_: initial states
             target_states _(list of array-like of shape (n, 1))_: target states
             tsave _(array-like of shape (nt,))_: times to be passed to sesolve
             params_to_optimize _(dict or array-like)_: parameters to optimize
                over that are used to define the Hamiltonian
             grape_type _(str)_: can be unitary, jumps, unitary_and_jumps. In the first
                case, just do sesolve dynamics. In the second just do mcsolve dynamics.
                In the third we can optimize over a combination of the unitary dynamics
                and jump dynamics. In this case we ignore the no-jump trajectories (redundant)
             jump_ops _(list of array-like of shape (n, n) or None)_: collapse operators
                if we are doing mcsolve dynamics
             target_states_traj _(list of array-like of shape (n, 1))_: target states for
                those initial states that experience jumps
             additional_drive_args _(into ot array-like)_: additional drive arguments
                that should be passed as an arg to timecallable that doesn't get optimized
                over. This is useful if you want to batch over mutliple Hamiltonian
                instances.
             filepath _(str)_: filepath of where to save optimization results
             optimizer _(optax.GradientTransformation)_: optax optimizer to use
                for gradient descent. Defaults to the Adam optimizer
             solver _(Solver)_: solver passed to sesolve
             options _(Options)_: options for grape optimization and sesolve integration
                relevant options include:
                    coherent, bool where if True we use a definition of fidelity
                    that includes relative phases, if not it ignores relative phases
                    epochs, int that is the maximum number of epochs to loop over
                    target_fidelity, float where the optimization terminates if the fidelity
                    if above this value
             init_params_to_save _(dict)_: initial parameters we want to save
        Returns:
            optimized parameters from the final timestep
        """
    initial_states = jnp.asarray(initial_states, dtype=cdtype())
    target_states = jnp.asarray(target_states, dtype=cdtype())
    if jump_ops is not None:
        target_states_traj = jnp.asarray(target_states_traj, dtype=cdtype())
    opt_state = optimizer.init(params_to_optimize)
    init_param_dict = options.__dict__ | {"tsave": tsave} | init_params_to_save
    print(f"saving results to {filepath}")
    try:  # trick for catching keyboard interrupt
        for epoch in range(options.epochs):
            epoch_start_time = time.time()
            params_to_optimize, opt_state, infids = step(
                params_to_optimize,
                opt_state,
                H_func,
                jump_ops,
                initial_states,
                target_states,
                target_states_traj,
                tsave,
                solver,
                options,
                optimizer,
                grape_type,
            )
            data_dict = {"infidelities": infids}
            save_and_print(
                filepath,
                data_dict,
                params_to_optimize,
                init_param_dict,
                epoch,
                epoch_start_time,
            )
            if all(infids < 1 - options.target_fidelity):
                print("target fidelity reached")
                break
        print(f"all results saved to {filepath}")
        return params_to_optimize
    except KeyboardInterrupt:
        print("terminated on keyboard interrupt")
        print(f"all results saved to {filepath}")
        return params_to_optimize


def save_and_print(
    filepath: str,
    data_dict: dict,
    params_to_optimize: ArrayLike | dict,
    init_param_dict: dict,
    epoch: int = 0,
    prev_time: float = 0.0,
):
    """saves the infidelities and optimal parameters obtained at each timestep"""
    infidelities = data_dict["infidelities"]
    if type(params_to_optimize) is dict:
        data_dict = data_dict | params_to_optimize
    else:
        data_dict["opt_params"] = params_to_optimize
    print(
        f"epoch: {epoch}, fids: {1 - infidelities},"
        f" elapsed_time: {np.around(time.time() - prev_time, decimals=3)} s"
    )
    if epoch != 0:
        append_to_h5(filepath, data_dict)
    else:
        write_to_h5_multi(filepath, data_dict, init_param_dict)


@partial(jax.jit, static_argnames=("solver", "options", "optimizer", "grape_type"))
def step(
    params_to_optimize,
    opt_state,
    H_func,
    jump_ops,
    initial_states,
    target_states,
    target_states_traj,
    tsave,
    solver,
    options,
    optimizer,
    grape_type,
):
    """calculate gradient of the loss and step updated parameters.
    We have has_aux=True because loss also returns the infidelities on the side
    (want to save those numbers as they give info on which pulse was best)"""
    grads, infids = jax.grad(loss, has_aux=True)(
        params_to_optimize,
        H_func,
        jump_ops,
        initial_states,
        target_states,
        target_states_traj,
        tsave,
        solver,
        options,
        grape_type,
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params_to_optimize = optax.apply_updates(params_to_optimize, updates)
    return params_to_optimize, opt_state, infids


def loss(
    params_to_optimize,
    H_func,
    jump_ops,
    initial_states,
    target_states,
    target_states_traj,
    tsave,
    solver,
    options,
    grape_type,
):
    """either calls sesolve or mcsolve or both depending on the type of simulation requested"""
    if options.coherent:
        infid_func = infidelity_coherent
    else:
        infid_func = infidelity_incoherent
    if grape_type == "unitary":
        infids = _unitary_infids(
            params_to_optimize,
            H_func,
            initial_states,
            target_states,
            tsave,
            solver,
            options,
            infid_func,
        )

    elif grape_type == "jumps":
        jump_no_jump_weights = jnp.array([1.0, 1.0])
        infids = _traj_infids(
            params_to_optimize,
            H_func,
            jump_ops,
            initial_states,
            target_states,
            target_states_traj,
            tsave,
            jump_no_jump_weights,
            solver,
            options,
            infid_func,
        )
    else:
        raise RuntimeError(f"grape_type must be unitary or jumps but got {grape_type}")
    return jnp.log(jnp.sum(infids)), infids


def _unitary_infids(
    params_to_optimize,
    H_func,
    initial_states,
    target_states,
    tsave,
    solver,
    options,
    infid_func,
):
    # update H using the same function but new parameters
    H_func = partial(H_func, drive_params=params_to_optimize)
    H = timecallable(H_func,)
    results = dq.sesolve(H, initial_states, tsave, solver=solver, options=options)
    infids = infid_func(results.final_state, target_states)
    if infids.ndim == 0:
        # for saving purposes, want this to be an Array as opposed to a float
        infids = infids[None]
    return infids


def _traj_infids(
    params_to_optimize,
    H_func,
    jump_ops,
    initial_states,
    target_states,
    target_states_jump,
    tsave,
    jump_no_jump_weights,
    solver,
    options,
    infid_func,
):
    H_func = partial(H_func, drive_params=params_to_optimize)
    H = timecallable(H_func, )
    mcsolve_results = dq.mcsolve(H, jump_ops, initial_states, tsave, solver=solver, options=options)
    final_jump_states = dq.unit(mcsolve_results.final_jump_states).swapaxes(-4, -3)
    final_no_jump_states = dq.unit(mcsolve_results.final_no_jump_state)
    infids_jump = infid_func(
        final_jump_states, target_states_jump
    )
    infids_jump_avg = jnp.mean(infids_jump)
    infids_no_jump = infid_func(
        final_no_jump_states, target_states
    )
    if infids_jump_avg.ndim == 0:
        # for saving purposes, want this to be an Array as opposed to a float
        infids_jump_avg = infids_jump_avg[None]
    if infids_no_jump.ndim == 0:
        infids_no_jump = infids_no_jump[None]
    infids = jnp.concatenate((jump_no_jump_weights[0] * infids_jump_avg,
                              jump_no_jump_weights[1] * infids_no_jump))
    return infids
