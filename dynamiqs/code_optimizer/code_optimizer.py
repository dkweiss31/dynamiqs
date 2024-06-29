from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
import numpy as np
import optax
from jaxtyping import ArrayLike

from dynamiqs import dag, unit, trace
from dynamiqs.utils.file_io import append_to_h5, write_to_h5_multi

from .._utils import cdtype

__all__ = ["code_optimizer"]


def code_optimizer(
    codewords,
    jump_ops=None,
    op_to_minimize=None,
    epochs=2000,
    target_infidelity=0.001,
    filepath: str = "tmp.h5py",
    optimizer: optax.GradientTransformation = optax.adam(0.1, b1=0.99, b2=0.99),
    init_params_to_save=None,
) -> ArrayLike:
    r"""Perform gradient descent to optimize codewords for QEC

        This function takes as input a list of initial codewords and a list
        of collapse operators and attempts to find codewords that satisfy the QEC condition
        It saves the parameters from every epoch and the associated fidelity
        in the file filepath

        Args:
             codewords _(dict or array-like)_: codewords to optimize
             jump_ops _(list of array-like of shape (n, n) or None)_: collapse operators
                if we are doing mcsolve dynamics
             epochs: number of epochs to optimize over
             filepath _(str)_: filepath of where to save optimization results
             optimizer _(optax.GradientTransformation)_: optax optimizer to use
                for gradient descent. Defaults to the Adam optimizer
             init_params_to_save _(dict)_: initial parameters we want to save
        Returns:
            optimized parameters from the final timestep
        """
    if init_params_to_save is None:
        init_params_to_save = {}
    codewords = jnp.asarray(codewords, dtype=cdtype())
    jump_ops = jnp.asarray(jump_ops, dtype=cdtype())
    op_to_minimize = jnp.asarray(op_to_minimize, dtype=cdtype())
    opt_state = optimizer.init(codewords)
    print(f"saving results to {filepath}")
    try:  # trick for catching keyboard interrupt
        for epoch in range(epochs):
            epoch_start_time = time.time()
            codewords, opt_state, infids = step(
                codewords,
                opt_state,
                jump_ops,
                op_to_minimize,
                optimizer,
            )
            data_dict = {"infidelities": infids}
            save_and_print(
                filepath,
                data_dict,
                codewords,
                init_params_to_save,
                epoch,
                epoch_start_time,
            )
            if all(infids < target_infidelity):
                print("target fidelity reached")
                break
        print(f"all results saved to {filepath}")
        return codewords
    except KeyboardInterrupt:
        print("terminated on keyboard interrupt")
        print(f"all results saved to {filepath}")
        return codewords


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
        f"epoch: {epoch}, cost: {infidelities},"
        f" elapsed_time: {np.around(time.time() - prev_time, decimals=3)} s"
    )
    if epoch != 0:
        append_to_h5(filepath, data_dict)
    else:
        write_to_h5_multi(filepath, data_dict, init_param_dict)


@partial(jax.jit, static_argnames=("optimizer",))
def step(
    codewords,
    opt_state,
    jump_ops,
    op_to_minimize,
    optimizer,
):
    """calculate gradient of the loss and step updated parameters.
    We have has_aux=True because loss also returns the infidelities on the side
    (want to save those numbers as they give info on which pulse was best)"""
    grads, infids = jax.grad(loss, has_aux=True)(
        codewords,
        jump_ops,
        op_to_minimize,
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    codewords = optax.apply_updates(codewords, updates)
    codewords = unit(codewords)
    return codewords, opt_state, infids


def loss(
    codewords,
    jump_ops,
    op_to_minimize,
):
    qec_mat = jnp.einsum(
        "wid,lij,kjm,smd->lkws",
        jnp.conj(codewords),
        dag(jump_ops),
        jump_ops,
        codewords
    )
    # penalize dependence on sigma
    qec_0 = qec_mat[:, :, 0, 0]
    qec_1 = qec_mat[:, :, 1, 1]
    infid = norm(qec_0 - qec_1, ord='fro')
    # penalize deviations from kroenecker delta
    for c_idx_1, codeword_1 in enumerate(codewords):
        for c_idx_2, codeword_2 in enumerate(codewords):
            if c_idx_1 != c_idx_2:
                infid += norm(qec_mat[:, :, c_idx_1, c_idx_2], ord='fro')
    density_matrices = jax.vmap(lambda cdwd: cdwd @ dag(cdwd))(codewords)
    max_mixed = jnp.sum(density_matrices, axis=0)
    minimize = jnp.real(trace(op_to_minimize @ max_mixed @ dag(op_to_minimize)))
    return infid + minimize, infid[None]
