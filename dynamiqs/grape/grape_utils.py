import jax
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft, fftfreq
from jax.random import PRNGKey


def generate_noise_trajectory(
        n_samples=1,
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
    low_pass_filter = lambda x, x0, order: 1. / jnp.sqrt(1 + jnp.abs(jnp.power(x / x0, order)))  # Low pass filter of arbitrary order, order=1 implies 1/f noise
    all_pass_filter = lambda x, x0, order: 1
    N = int(sample_rate * t_max) + 1  # total number of time points
    t_list = jnp.linspace(0, t_max, N)  # Time list
    dt = jnp.mean(jnp.diff(t_list))

    # First, generate a time domain trajectory that is unfiltered
    # Frequency shift at each sample of time
    freq_shifts = (
        jnp.sqrt(relative_PSD_strength * sample_rate)
        * jax.random.normal(key, shape=(N, n_samples,),)
    )
    # Convert to frequency domain
    freq_y_vals = fft(freq_shifts, axis=0)
    freq_x_vals = fftfreq(N, d=dt)
    # Filter as required
    freq_filter = all_pass_filter if white else low_pass_filter
    freq_y_vals = jnp.einsum("ij,i->ij", freq_y_vals, freq_filter(freq_x_vals, f0, order=1))
    # Record PSD for plotting purposes
    freq_psd_vals = jnp.abs(freq_y_vals[:N // 2]) ** 2  # Power spectral density is positive (or negative) frequency half of FFT, squared
    # Convert back to time domain
    filtered_freq_shifts = jnp.real(ifft(freq_y_vals, axis=0))  # Filtered freq shift at each time
    filtered_trajectory = jnp.cumsum(filtered_freq_shifts, axis=0) * dt  # Filtered total phase shift vs time
    return t_list, filtered_freq_shifts, filtered_trajectory, freq_x_vals[:N // 2], freq_psd_vals
