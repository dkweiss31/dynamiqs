import matplotlib.pyplot as plt

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from dynamiqs import timecallable, grape, GRAPEOptions
from quantum_utils import generate_file_path, extract_info_from_h5
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

run_grape = True
read_filepath = "out/00021_test_grape_drag.h5py"

N = 8
Kerr = 2.0 * jnp.pi * 0.130
a = dq.destroy(N)
H0 = -0.5 * Kerr * dq.dag(a) @ dq.dag(a) @ a @ a
H1 = [a + dq.dag(a), -1j * (a - dq.dag(a))]
initial_states = [dq.basis(N, 0), dq.basis(N, 1)]
target_states = [dq.basis(N, 1), dq.basis(N, 0)]
dt = 10.0
time = 100.0
tsave = jnp.linspace(0, time, int(time // dt))
N_ringup = 1
scale = 0.01
bandwidth = 0.5
N_cutoff = int(bandwidth * len(tsave) / 2)
rng = np.random.default_rng(423456)
#grape_params_1 = -2.0 * scale * rng.random((len(tsave))) + scale
grape_params_1 = 0.01 * np.exp(-0.5 * (tsave - tsave[-1]/2)**2 / (tsave[-1]/4)**2)
#grape_params_1 = 0.3 * (np.pi / 2) / tsave[-1] * np.ones(len(tsave))  # pi/2 pulse
grape_params_2 = np.zeros(len(tsave)) + 1e-8
grape_params = np.stack((grape_params_1, grape_params_2))
save_filepath = generate_file_path("h5py", "test_grape_drag", "out")
options = GRAPEOptions(N_multistart=1, target_fidelity=0.9995,
                       learning_rate=0.001, epochs=1000, coherent=True)


def ringup_envelope(N_ringup, times):
    N_blocks = times.shape[-1]
    ringup_env = (1 - jnp.cos(jnp.linspace(0.0, np.pi, N_ringup))) / 2
    envelope = jnp.concatenate([
        ringup_env,
        jnp.ones([N_blocks - 2 * N_ringup]),
        jnp.flip(ringup_env)
    ])
    return envelope


envelope = ringup_envelope(1, tsave)


def H_func(t, params):
    H = H0
    for drive_idx, opt_params in enumerate(params):
        opt_params_w_env = jnp.einsum("k,k->k", envelope, opt_params)
        spline = InterpolatedUnivariateSpline(tsave, opt_params_w_env, endpoints="natural")
        H = H + H1[drive_idx] * spline(t)
    return H


H = timecallable(H_func, args=(grape_params,))

if run_grape:
    grape(H,
        initial_states=initial_states,
        target_states=target_states,
        tsave=tsave,
        params_to_optimize=grape_params,
        filepath=save_filepath,
        options=options
    )
    read_filepath = save_filepath
data, params = extract_info_from_h5(read_filepath)
opt_params = data["opt_params"][-1]
H = timecallable(H_func, args=(opt_params,))
exp_ops = [dq.basis(N, i) @ dq.tobra(dq.basis(N, i)) for i in range(N)]
res = dq.sesolve(H, initial_states, tsave, exp_ops=exp_ops)
fig, ax = plt.subplots()
for idx in range(N):
    plt.plot(tsave, res.expects[0, idx], label=f"{idx}")
ax.legend()
plt.show()
fig, ax = plt.subplots()
plt.plot(tsave, opt_params[0], label="H1_1")
plt.plot(tsave, opt_params[1], label="H1_2")
ax.legend()
plt.show()
print(0)

