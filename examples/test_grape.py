import matplotlib.pyplot as plt
import optax
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from dynamiqs import timecallable, grape, Options
from dynamiqs.utils.file_io import generate_file_path

N = 4
Kerr = 2.0 * jnp.pi * 0.130
a = dq.destroy(N)
H0 = -0.5 * Kerr * dq.dag(a) @ dq.dag(a) @ a @ a
H1 = [a + dq.dag(a), 1j * (a - dq.dag(a))]
initial_states = [dq.basis(N, 0), dq.basis(N, 1)]
target_states = [dq.basis(N, 1), dq.basis(N, 0)]
dt = 5.0
time = 50.0
ntimes = int(time // dt) + 1
tsave = jnp.linspace(0, time, ntimes)
grape_params = jnp.stack((0.01 * jnp.ones(len(tsave)), 0.001 * jnp.ones(len(tsave))))
save_filepath = generate_file_path("h5py", "test_grape_drag", "out")
options = Options(target_fidelity=0.9995, epochs=1000, coherent=True)
optimizer = optax.adam(learning_rate=0.001, b1=0.999, b2=0.999)


def H_func(t, params_to_optimize):
    H = H0
    for drive_idx, drive_params in enumerate(params_to_optimize):
        envelope = jnp.concatenate((jnp.array([0.0]), jnp.ones(ntimes - 2), jnp.array([0.0])))
        drive_params_env = envelope * drive_params
        spline = InterpolatedUnivariateSpline(tsave, drive_params_env, endpoints="natural")
        H = H + H1[drive_idx] * spline(t)
    return H


H = timecallable(H_func, args=(grape_params,))
opt_params = grape(H, initial_states=initial_states, target_states=target_states,
      tsave=tsave, params_to_optimize=grape_params, filepath=save_filepath,
      optimizer=optimizer, options=options
      )
H = timecallable(H_func, args=(opt_params,))
exp_ops = [dq.basis(N, i) @ dq.tobra(dq.basis(N, i)) for i in range(N)]
res = dq.sesolve(H, initial_states, tsave, exp_ops=exp_ops)

fig, ax = plt.subplots()
envelope = jnp.concatenate((jnp.array([0.0]), jnp.ones(ntimes - 2), jnp.array([0.0])))
I_spline = InterpolatedUnivariateSpline(tsave, envelope * opt_params[0], endpoints="natural")
Q_spline = InterpolatedUnivariateSpline(tsave, envelope * opt_params[1], endpoints="natural")
finer_times = np.linspace(0.0, time, 101)
plt.plot(finer_times, I_spline(finer_times), label="H1_1")
plt.plot(finer_times, Q_spline(finer_times), label="H1_2")
ax.legend()
plt.show()
fig, ax = plt.subplots()
for idx in range(N):
    plt.plot(tsave, res.expects[0, idx], label=f"{idx}")
ax.legend()
plt.show()
fig, ax = plt.subplots()
for idx in range(N):
    plt.plot(tsave, res.expects[1, idx], label=f"{idx}")
ax.legend()
plt.show()
