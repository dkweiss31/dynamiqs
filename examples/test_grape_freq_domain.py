import jax
import matplotlib.pyplot as plt

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from dynamiqs import timecallable, grape, GRAPEOptions
from quantum_utils import generate_file_path, extract_info_from_h5
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from dynamiqs.solver import Dopri5, Tsit5

import qutip as qt


N = 8
Kerr = 2.0 * jnp.pi * 0.130
a = dq.destroy(N)
H0 = -0.5 * Kerr * dq.dag(a) @ dq.dag(a) @ a @ a
H1 = [a + dq.dag(a), 1j * (a - dq.dag(a))]
initial_states = [dq.basis(N, 0), dq.basis(N, 1)]
target_states = [dq.basis(N, 1), dq.basis(N, 0)]
dt = 1.0
time = 50.0
tsave = jnp.linspace(0, time, int(time // dt) + 1)
N_ringup = 5
scale = 0.001
bandwidth = 0.2
N_cutoff = int(bandwidth * len(tsave) / 2)
rng_key = PRNGKey(4223223325)
save_filepath = generate_file_path("h5py", "test_grape_drag_freq", "out")
options = GRAPEOptions(target_fidelity=0.9995,
                       learning_rate=0.01, epochs=10000, coherent=True)


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


def randomize_and_set_vars(key=None, scale=0.01):
    """randomize initial pulses"""
    if key is None:
        key = PRNGKey(42)
    opt_params = {}
    for k in range(len(H1) // 2):
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


def H_func(t, drive_params):
    H_tot = H0
    for drive_idx in range(len(H1) // 2):
        control_signal = get_IQ_time_series(drive_idx, drive_params, tsave, N_ringup, N_cutoff)
        I_control_idx = jnp.real(control_signal)
        Q_control_idx = jnp.imag(control_signal)
        I_spline = InterpolatedUnivariateSpline(tsave, I_control_idx, endpoints="natural")
        Q_spline = InterpolatedUnivariateSpline(tsave, Q_control_idx, endpoints="natural")
        H_tot = H_tot + H1[2 * drive_idx] * I_spline(t) + H1[2 * drive_idx + 1] * Q_spline(t)
    return H_tot


run_grape = True
read_filepath = "out/00051_test_grape_drag_freq.h5py"

grape_params = randomize_and_set_vars(rng_key, scale)
H = timecallable(H_func, args=(grape_params,))
solver = Dopri5()


if run_grape:
    grape(
        H,
        initial_states=initial_states,
        target_states=target_states,
        tsave=tsave,
        params_to_optimize=grape_params,
        filepath=save_filepath,
        solver=Tsit5(),
        options=options
    )
    read_filepath = save_filepath
data, params = extract_info_from_h5(read_filepath)
# opt_params = data["opt_params"][-1]
data_last = dict((k, v[-1]) for k, v in data.items())
H = timecallable(H_func, args=(data_last,))
exp_ops = [dq.basis(N, i) @ dq.tobra(dq.basis(N, i)) for i in range(N)]
res = dq.sesolve(H, initial_states, tsave, exp_ops=exp_ops)
I_Q = get_IQ_time_series(0, data_last, tsave, N_ringup, N_cutoff)

I_spline = InterpolatedUnivariateSpline(tsave, np.real(I_Q), endpoints="natural")
Q_spline = InterpolatedUnivariateSpline(tsave, np.imag(I_Q), endpoints="natural")
# test with qutip
qt_a = qt.destroy(N)
qt_H0 = -0.5 * Kerr * qt_a.dag() * qt_a.dag() * qt_a * qt_a
qt_H1 = [qt_a + qt_a.dag(), 1j * (qt_a - qt_a.dag())]
qt_H = [qt_H0, [qt_H1[0], lambda t, args: I_spline(t)], [qt_H1[1], lambda t, args: Q_spline(t)]]
qt_times = np.linspace(0.0, time, int(time // dt) + 1)
res_qt = qt.sesolve(qt_H, qt.basis(N, 0), qt_times,
                    e_ops=[qt.basis(N, i) * qt.basis(N, i).dag() for i in range(N)])

fig, ax = plt.subplots()
plt.plot(tsave, np.real(I_Q), label="H1_1")
plt.plot(tsave, np.imag(I_Q), label="H1_2")
ax.legend()
plt.show()
fig, ax = plt.subplots()
for idx in range(N):
    plt.plot(tsave, res.expects[0, idx], label=f"{idx}")
    plt.plot(tsave, res_qt.expect[idx], ls='--', label=f"qt {idx}")
ax.legend()
plt.show()
fig, ax = plt.subplots()
for idx in range(N):
    plt.plot(tsave, res.expects[1, idx], label=f"{idx}")
ax.legend()
plt.show()
