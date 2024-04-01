import jax
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft, fftfreq
from jax.random import PRNGKey
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


def generate_noise_trajectory(
        sample_rate=1,
        t_max=200,
        relative_PSD_strength=(1e-3) ** 2,
        f0=1e-2,
        white=False,
        rng_seed=42,
):
    """"
    Provide sample rate (in us^-1), max time (in us), relative_PSD_strength (in (us^-2)/sample_rate),
    cutoff frequency f0 (in us^-1) and whether noise is white
    Noise will be sampled at sample_rate, generating gaussian (or 1/f filtered) frequency shifts,
    with std-deviation given by sqrt(relative_PSD_strength * sample_rate)
    Output is time list, a single trajectory over the time list,
              frequency list, and the effective power spectral density in this instance
    """
    key = PRNGKey(rng_seed)
    low_pass_filter = lambda x, x0, order: 1. / jnp.sqrt(1 + jnp.power(x / x0, 2 * order))  # Low pass filter of arbitrary order, order=1 implies 1/f noise
    all_pass_filter = lambda x, x0, order: 1
    N = int(sample_rate * t_max) + 1  # total number of time points
    t_list = jnp.linspace(0, t_max, N)  # Time list
    dt = jnp.mean(jnp.diff(t_list))

    # First, generate a time domain trajectory that is unfiltered
    # Frequency shift at each sample of time
    freq_shifts = (
        jnp.sqrt(relative_PSD_strength * sample_rate)
        * jax.random.uniform(key, shape=(N,), minval=0., maxval=1.)
    )
    # Convert to frequency domain
    freq_y_vals = fft(freq_shifts)
    freq_x_vals = fftfreq(len(freq_shifts), d=dt)
    # Filter as required
    freq_filter = all_pass_filter if white else low_pass_filter
    freq_y_vals = jnp.multiply(freq_y_vals, freq_filter(freq_x_vals, f0, order=1))
    # Record PSD for plotting purposes
    freq_psd_vals = jnp.abs(freq_y_vals[:N // 2]) ** 2  # Power spectral density is positive (or negative) frequency half of FFT, squared
    # Convert back to time domain
    filtered_freq_shifts = jnp.real(ifft(freq_y_vals))  # Filtered freq shift at each time
    filtered_trajectory = jnp.cumsum(filtered_freq_shifts) * dt  # Filtered total phase shift vs time
    return t_list, filtered_freq_shifts, filtered_trajectory, freq_x_vals[:N // 2], freq_psd_vals


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
