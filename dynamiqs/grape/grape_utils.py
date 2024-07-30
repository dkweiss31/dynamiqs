import jax
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft, fftfreq
from jax.random import PRNGKey
from dynamiqs import mesolve, sesolve, dag
from dynamiqs.solver import Tsit5
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from dynamiqs import Options


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


def T2_echo_experiment(H, init_state, delay_times, X_op, readout_proj,
                       atol=1e-8, rtol=1e-8, max_steps=1_000_000):
    final_probs = np.zeros_like(delay_times)
    for t_idx, delay_time in enumerate(delay_times[1:]):
        first_half_res = sesolve(
            H,
            init_state,
            (0, delay_time // 2),
            solver=Tsit5(atol=atol, rtol=rtol, max_steps=max_steps),
            options=Options(save_states=False)
        )
        state_after_pi = X_op @ first_half_res.states[-1]
        second_half_res = sesolve(
            H,
            state_after_pi,
            (delay_time // 2, delay_time),
            solver=Tsit5(atol=atol, rtol=rtol, max_steps=max_steps),
            options=Options(cartesian_batching=False),
        )
        final_state = X_op @ second_half_res.states[..., -1, :, :]
        pop = dag(final_state) @ readout_proj @ final_state
        final_probs[t_idx + 1] = jnp.average(pop)
    return final_probs


def T2_Ramsey_experiment(H, init_state, delay_times, readout_proj,
                         atol=1e-8, rtol=1e-8, max_steps=1_000_000):
    res = sesolve(H, init_state, delay_times, exp_ops=[readout_proj,],
                  solver=Tsit5(atol=atol, rtol=rtol, max_steps=max_steps))
    return jnp.average(res.expects, axis=0)[0]  # average over batch dimension


def T2_func_exp(t, t2, a, b):
    return a * jnp.exp(-t / t2) + b

def T2_func_Gauss(t, t2, a, b):
    return a * jnp.exp(-0.5 * t**2 / t2**2) + b


def extract_gammaphi(
    ramsey_result,
    delay_times,
    p0=(6 * 10**3, 1.0, 0.0),
    plot=True,
    type="exp"
):
    if type == "exp":
        T2_func = T2_func_exp
    elif type == "gauss":
        T2_func = T2_func_Gauss
    else:
        raise ValueError("type needs to be exp or gauss")
    popt_T2, pcov_T2 = curve_fit(
        T2_func,
        delay_times,
        ramsey_result,
        p0=p0,
        maxfev=6000,
        bounds=((100, -1.0, -1.0), (10 ** 15, 1.0, 1.0)),
    )
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(delay_times, ramsey_result, "o")
        plot_times = jnp.linspace(0.0, delay_times[-1], 2000)
        ax.plot(plot_times, T2_func(plot_times, *popt_T2), linestyle="-")
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel(r"$P(|+\rangle)$", fontsize=12)
        ax.set_xlabel("time [ns]", fontsize=12)
        plt.show()
    print("popt: ", popt_T2)
    print("pcov: ", pcov_T2)
    print("condition_num: ", jnp.linalg.cond(pcov_T2))
    return (1 / popt_T2[0]) * 10**6 / (2 * jnp.pi)
