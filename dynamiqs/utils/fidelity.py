from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    'infidelity_coherent',
    'infidelity_incoherent',
]

from dynamiqs import isket, isbra, toket


def _overlaps(computed_states, target_states):
    if isbra(computed_states):
        computed_states = toket(computed_states)
    if isbra(target_states):
        target_states = toket(target_states)
    # s: batch over initial states, i: hilbert dim, d: size 1
    return jnp.einsum(
        "...sid,...sid->...s", jnp.conj(target_states), computed_states
    )


def infidelity_coherent(computed_states, target_states):
    """compute coherent definition of the fidelity allowing for batching.
    assumption is that the initial states to average over are the second to
    last index, and the last index is hilbert dim"""
    overlaps = _overlaps(computed_states, target_states)
    overlaps_avg = jnp.mean(overlaps, axis=-1)
    fids = jnp.abs(overlaps_avg * jnp.conj(overlaps_avg))
    return 1 - fids


def infidelity_incoherent(computed_states, target_states):
    """as above in fidelity_incoherent, but now average over the initial
    states after squaring the overlaps, erasing phase information"""
    overlaps = _overlaps(computed_states, target_states)
    overlaps_sq = jnp.abs(overlaps * jnp.conj(overlaps))
    fids = jnp.mean(overlaps_sq, axis=-1)
    return 1 - fids
