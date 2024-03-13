from __future__ import annotations

import jax
from functools import partial
import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpy as np
import optax

from .._utils import cdtype
import time
from ..solver import Solver
from dynamiqs.utils.fidelity import infidelity_coherent, infidelity_incoherent
import dynamiqs as dq
from dynamiqs import timecallable, GRAPEOptions, TimeArray
from quantum_utils import write_to_h5_multi, append_to_h5

__all__ = ['grape']


def grape(
    H: ArrayLike | TimeArray,
    initial_states: ArrayLike,
    target_states: ArrayLike,
    tsave: ArrayLike,
    params_to_optimize: ArrayLike,
    filepath: str,
    solver: Solver,
    options: GRAPEOptions = GRAPEOptions(),
):
    initial_states = jnp.asarray(initial_states, dtype=cdtype())
    target_states = jnp.asarray(target_states, dtype=cdtype())
    optimizer = optax.adam(options.learning_rate, b1=0.9999, b2=0.9999)
    opt_state = optimizer.init(params_to_optimize)
    param_dict = options.__dict__ | {"tsave": tsave}
    print(f"saving results to {filepath}")
    try:  # trick for catching keyboard interrupt
        for epoch in range(options.epochs):
            epoch_start_time = time.time()
            params_to_optimize, opt_state, infids = step(
                params_to_optimize, opt_state, H, initial_states,
                target_states, tsave, solver, options, optimizer
            )
            data_dict = {"infidelities": infids}
            if type(params_to_optimize) is dict:
                data_dict = data_dict | params_to_optimize
            else:
                data_dict["opt_params"] = params_to_optimize
            save_and_print(filepath, data_dict, param_dict, epoch, epoch_start_time)
            if any(infids < 1 - options.target_fidelity):
                print("target fidelity reached")
                break
        print(f"all results saved to {filepath}")
    except KeyboardInterrupt:
        print("terminated on keyboard interrupt")
        print(f"all results saved to {filepath}")


def save_and_print(filepath, data_dict: dict, param_dict, epoch=0, prev_time=0.0):
    infidelities = data_dict["infidelities"]
    print(f"epoch: {epoch}, fids: {1 - infidelities},"
          f" elapsed_time: {np.around(time.time() - prev_time, decimals=3)} s")
    if epoch != 0:
        append_to_h5(filepath, data_dict)
    else:
        write_to_h5_multi(filepath, data_dict, param_dict)


@partial(jax.jit, static_argnames=('solver', 'options', 'optimizer'))
def step(params_to_optimize, opt_state, H, initial_states,
         target_states, tsave, solver, options, optimizer,
         ):
    grads, infids = jax.grad(loss, has_aux=True)(params_to_optimize, H, initial_states,
                                                 target_states, tsave, solver, options)
    updates, opt_state = optimizer.update(grads, opt_state)
    params_to_optimize = optax.apply_updates(params_to_optimize, updates)
    return params_to_optimize, opt_state, infids


def loss(params_to_optimize, H, initial_states, target_states, tsave, solver, options):
    # update H using the same function but new parameters
    H = timecallable(H.f, args=(params_to_optimize,))
    result = dq.sesolve(H, initial_states, tsave, solver=solver)
    # result.states has shape (bH?, bpsi?, nt, n, 1) and we want the states at the final time
    final_states = result.states[..., -1, :, :]
    if options.coherent:
        infids = infidelity_coherent(final_states, target_states)
    else:
        infids = infidelity_incoherent(final_states, target_states)
    if infids.ndim == 0:
        infids = infids[None]
    return jnp.sum(jnp.log(infids)), infids
