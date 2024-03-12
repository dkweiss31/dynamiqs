from __future__ import annotations

import jax
from functools import partial
from jax.random import PRNGKey
import jax.numpy as jnp
import copy
import numpy as np
import optax
from .._utils import cdtype
import time
from dynamiqs.utils.fidelity import infidelity_coherent, infidelity_incoherent
import dynamiqs as dq
from dynamiqs import timecallable
from quantum_utils import write_to_h5_multi, append_to_h5

__all__ = ['GRAPE']


class GRAPE:
    def __init__(
        self,
        H_func=None,
        initial_states=None,
        target_states=None,
        times=None,
        jump_ops=None,
        N_multistart=1,
        target_fidelity=0.9995,
        grape_params=None,
        learning_rate=0.3,
        epochs=1000,
        coherent=True,
        filepath="tmp.h5py",
        key: PRNGKey = PRNGKey(42)
    ):
        self.H_func = H_func
        self.initial_states = jnp.asarray(initial_states, dtype=cdtype())
        self.target_states = jnp.asarray(target_states, dtype=cdtype())
        self.times = times
        self.jump_ops = jump_ops
        self.N_multistart = N_multistart
        self.target_fidelity = target_fidelity
        self.grape_params = grape_params
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coherent = coherent
        self.filepath = filepath
        self.key = key
        self._init_attrs = set(self.__dict__.keys())
        self._init_params = copy.deepcopy(self.__dict__)
        self.optimizer = optax.adam(self.learning_rate, b1=0.9999, b2=0.9999)
        self.opt_state = self.optimizer.init(self.grape_params)
        if self.coherent:
            self.compute_infidelity = infidelity_coherent
        else:
            self.compute_infidelity = infidelity_incoherent

    def save_and_print(self, additional_data: dict, epoch=0, prev_time=0.0):
        data_dict = {"opt_params": self.grape_params} | additional_data
        infidelities = additional_data["infidelities"]
        print(f"epoch: {epoch}, fids: {1 - infidelities},"
              f" elapsed_time: {np.around(time.time() - prev_time, decimals=3)} s")
        if epoch != 0:
            append_to_h5(self.filepath, data_dict)
        else:
            write_to_h5_multi(self.filepath, data_dict, self._init_params)

    @staticmethod
    @partial(jax.jit, static_argnames=('loss_fun', 'optimizer'))
    def step(grape_params, opt_state, loss_fun, optimizer):
        # note explicitly passing self.grape_params and
        # self.opt_state as arguments to loss
        grads, infids = jax.grad(loss_fun, has_aux=True)(grape_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        grape_params = optax.apply_updates(grape_params, updates)
        return grape_params, opt_state, infids

    def loss(self, grape_params):
        H = timecallable(self.H_func, args=(grape_params,))
        res = dq.sesolve(H, self.initial_states, self.times)
        # res.states has shape (bH?, bpsi?, nt, n, 1) and we want the states at the final time
        final_states = res.states[..., -1, :, :]
        infids = self.compute_infidelity(final_states, self.target_states)
        if infids.ndim == 0:
            infids = infids[None]
        return jnp.sum(jnp.log(infids)) / self.N_multistart, infids

    def run(self):
        """run the optimization"""
        print(f"saving results to {self.filepath}")
        try:  # trick for catching keyboard interrupt
            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                grape_params, opt_state, infids = self.step(
                    self.grape_params, self.opt_state, self.loss, self.optimizer
                )
                self.grape_params = grape_params
                self.opt_state = opt_state
                self.save_and_print({"infidelities": infids}, epoch, epoch_start_time)
                if any(infids < 1 - self.target_fidelity):
                    print("target fidelity reached")
                    break
        except KeyboardInterrupt:
            print("terminated on keyboard interrupt")
            print(f"all results saved to {self.filepath}")
