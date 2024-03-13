import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


def ringup_envelope(N_ringup, times):
    N_blocks = times.shape[-1]
    ringup_env = (1 - jnp.cos(jnp.linspace(0.0, np.pi, N_ringup))) / 2
    envelope = jnp.concatenate([
        ringup_env,
        jnp.ones([N_blocks - 2 * N_ringup]),
        jnp.flip(ringup_env)
    ])
    return envelope


def get_IQ_time_series(control_idx, opt_params, times, N_ringup, N_cutoff):
    """ported almost directly from the tensorflow code. beautiful idea to optimize over
    the frequency components rather than the time points."""
    N_blocks = len(times)
    I_DC = opt_params[f"I_DC{control_idx}"]
    I_real = opt_params[f"I_real{control_idx}"]
    I_imag = opt_params[f"I_imag{control_idx}"]
    Q_DC = opt_params[f"Q_DC{control_idx}"]
    Q_real = opt_params[f"Q_real{control_idx}"]
    Q_imag = opt_params[f"Q_imag{control_idx}"]
    I_comps = jax.lax.complex(I_real, I_imag)
    Q_comps = jax.lax.complex(Q_real, Q_imag)
    DC_comps = jax.lax.complex(I_DC, Q_DC)
    positive_comps = I_comps + 1j * Q_comps
    negative_comps = (jnp.conj(jnp.flip(I_comps, axis=0))
                      + 1j * jnp.conj(jnp.flip(Q_comps, axis=0)))
    zeros = jnp.zeros((N_blocks - 1 - 2 * N_cutoff), dtype=jnp.complex64)
    freq_comps = jnp.concatenate([DC_comps, positive_comps, zeros, negative_comps], axis=0)
    envelope = ringup_envelope(N_ringup, times)
    signal = jnp.fft.ifft(freq_comps, axis=0)
    return jnp.einsum("k,k...->k...", envelope, signal)


def randomize_and_set_IQ_vars(N_cutoff, N_drives, key=None, scale=0.01):
    """randomize initial pulses"""
    if key is None:
        key = PRNGKey(42)
    opt_params = {}
    for k in range(N_drives // 2):
        for quadrature in ("I", "Q"):
            keys = jax.random.split(key, 4)
            quad_DC = - 2.0 * scale * jax.random.uniform(keys[0], (1, ),) + scale
            quad_real = - 2.0 * scale * jax.random.uniform(keys[1], (N_cutoff, ),) + scale
            quad_imag = - 2.0 * scale * jax.random.uniform(keys[2], (N_cutoff, ),) + scale
            opt_params[f"{quadrature}_DC{k}"] = quad_DC
            opt_params[f"{quadrature}_real{k}"] = quad_real
            opt_params[f"{quadrature}_imag{k}"] = quad_imag
            key = keys[3]
    return opt_params


def H_evaluated_at_t(H0, H1s, tsave, N_ringup, N_cutoff, t, drive_params):
    """this function will eventually be passed to partial to have the appropriate signature
    of a CallableTimeArray:
    H_func = Partial(H_evaluated_at_t, H0, H1, tsave, N_ringup, N_cutoff)
    H = timecallable(H_func, args=(drive_params,))
    """
    H_tot = H0
    for drive_idx in range(len(H1s) // 2):
        control_signal = get_IQ_time_series(drive_idx, drive_params, tsave, N_ringup, N_cutoff)
        I_control_idx = jnp.real(control_signal)
        Q_control_idx = jnp.imag(control_signal)
        I_spline = InterpolatedUnivariateSpline(tsave, I_control_idx, endpoints="natural")
        Q_spline = InterpolatedUnivariateSpline(tsave, Q_control_idx, endpoints="natural")
        H_tot = H_tot + H1s[2 * drive_idx] * I_spline(t) + H1s[2 * drive_idx + 1] * Q_spline(t)
    return H_tot
