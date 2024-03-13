import matplotlib.pyplot as plt

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from dynamiqs import timecallable, grape, Options
from dynamiqs.grape.grape_utils import H_evaluated_at_t, get_IQ_time_series, randomize_and_set_IQ_vars
from jax.tree_util import Partial
from quantum_utils import generate_file_path, extract_info_from_h5
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from dynamiqs.solver import Dopri5, Tsit5
import optax



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
options = Options(target_fidelity=0.9995, epochs=10000, coherent=True)
optimizer = optax.adam(learning_rate=0.01, b1=0.999, b2=0.999)

run_grape = True
read_filepath = "out/00051_test_grape_drag_freq.h5py"

grape_params = randomize_and_set_IQ_vars(N_cutoff, len(H1), rng_key, scale)
H_func = Partial(H_evaluated_at_t, H0, H1, tsave, N_ringup, N_cutoff)
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
        optimizer=optimizer,
        solver=Tsit5(),
        options=options
    )
    read_filepath = save_filepath
data, params = extract_info_from_h5(read_filepath)
data_last = dict((k, v[-1]) for k, v in data.items())
H = timecallable(H_func, args=(data_last,))
exp_ops = [dq.basis(N, i) @ dq.tobra(dq.basis(N, i)) for i in range(N)]
res = dq.sesolve(H, initial_states, tsave, exp_ops=exp_ops)
I_Q = get_IQ_time_series(0, data_last, tsave, N_ringup, N_cutoff)

I_spline = InterpolatedUnivariateSpline(tsave, np.real(I_Q), endpoints="natural")
Q_spline = InterpolatedUnivariateSpline(tsave, np.imag(I_Q), endpoints="natural")

fig, ax = plt.subplots()
plt.plot(tsave, np.real(I_Q), label="H1_1")
plt.plot(tsave, np.imag(I_Q), label="H1_2")
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
