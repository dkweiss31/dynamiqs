from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    'infidelity_coherent',
    'infidelity_incoherent',
    'forbidden_population',
    'all_X_Y_Z_states',
]

from dynamiqs import isket, isbra, toket, unit


def _overlaps(computed_states, target_states):
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


def forbidden_population(computed_states, forbidden_states):
    forbidden_ovlp = jnp.einsum(
        "...sid,sbid->...b",
        computed_states[..., 0: len(forbidden_states), :, :],
        forbidden_states
    )
    # average over everything
    return jnp.mean(jnp.abs(forbidden_ovlp * jnp.conj(forbidden_ovlp)))


def all_X_Y_Z_states(basis_states):
    all_states = []
    for idx, state in enumerate(basis_states):
        all_states.append(state)
    for idx_1, state_1 in enumerate(basis_states):
        for idx_2, state_2 in enumerate(basis_states):
            if idx_2 > idx_1:
                all_states.append(unit(state_1 + state_2))
                all_states.append(unit(state_1 + 1j * state_2))
    return all_states
