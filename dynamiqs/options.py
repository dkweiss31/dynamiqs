from __future__ import annotations

import equinox as eqx
import jax
from jax import Array
from jaxtyping import PyTree, Scalar
from jax.random import PRNGKey

__all__ = ['Options', 'GRAPEOptions']


class GRAPEOptions(eqx.Module):
    target_fidelity: float
    learning_rate: float
    epochs: int
    coherent: bool

    def __init__(
        self,
        target_fidelity: float = 0.9995,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        coherent: bool = True,
    ):
        """ Generic options for GRAPE

        Args:
            N_multistart: number of batches to optimize over simultaneously
            target_fidelity: if this fidelity is reached, stop optimization
            learning_rate: learning rate for Adam optimizer
            epochs: number of epochs to loop over
            coherent: If true, use coherent definition of the fidelity which
                      accounts for relative phases. If not, use incoherent
                      definition
        """
        self.target_fidelity = target_fidelity
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coherent = coherent


class Options(eqx.Module):
    save_states: bool
    verbose: bool
    cartesian_batching: bool
    t0: Scalar | None
    save_extra: callable[[Array], PyTree] | None

    def __init__(
        self,
        save_states: bool = True,
        verbose: bool = True,
        cartesian_batching: bool = True,
        t0: Scalar | None = None,
        save_extra: callable[[Array], PyTree] | None = None,
    ):
        """Generic options for the quantum solvers.

        Args:
            save_states: If `True`, the state is saved at every time in `tsave`,
                otherwise only the final state is returned.
            verbose: If `True`, print information about the integration, otherwise
                nothing is printed.
            cartesian_batching: If `True`, batched arguments are treated as separated
                batch dimensions, otherwise the batching is performed over a single
                shared batched dimension.
            t0: Initial time. If `None`, defaults to the first time in `tsave`.
            save_extra _(function, optional)_: A function with signature
                `f(Array) -> PyTree` that takes a state as input and returns a PyTree.
                This can be used to save additional arbitrary data during the
                integration.
        """
        self.save_states = save_states
        self.verbose = verbose
        self.cartesian_batching = cartesian_batching
        self.t0 = t0
        if save_extra is not None:
            # use `jax.tree_util.Partial` to make `save_extra` a valid Pytree
            self.save_extra = jax.tree_util.Partial(save_extra)
        else:
            self.save_extra = save_extra
