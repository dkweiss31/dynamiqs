from functools import partial

import jax.numpy as jnp
import jax
import numpy as np
import optax

from dynamiqs import Options, grape, dag, basis, destroy, sesolve, timecallable
import diffrax as dx


dim = 4
optimizer = optax.adam(learning_rate=0.0005)
ntimes = 100
tsave = jnp.linspace(0, 100, 100)
options = Options(
    save_states=False,
)
a = destroy(dim)
H0 = -0.5 * 0.1 * 2.0 * jnp.pi * dag(a) @ dag(a) @ a @ a
H1 = [a + dag(a), 1j * (a - dag(a))]

initial_states = [basis(dim, 0), basis(dim, 1)]
final_states = [basis(dim, 1), basis(dim, 0)]

rng = np.random.default_rng(1234)
init_drive_params = 2.0 * jnp.pi * (-2.0 * 0.1 * rng.random((len(H1), ntimes)) + 0.1)


def _drive_at_time(t, drive_param):
    drive_coeffs = dx.backward_hermite_coefficients(tsave, drive_param)
    drive_spline = dx.CubicInterpolation(tsave, drive_coeffs)
    return drive_spline.evaluate(t)


def H_func(t, drive_params,):
    H = H0
    for drive_idx in range(len(H1)):
        drive_amp = _drive_at_time(t, drive_params[drive_idx])
        H = H + drive_amp * H1[drive_idx]
    return H


H_func = partial(H_func, drive_params=init_drive_params)
H = timecallable(H_func, )

res = sesolve(H, initial_states, tsave, options=options)

H_tc = jax.tree_util.Partial(H_func)
opt_params = grape(
    H_tc,
    initial_states=initial_states,
    target_states=final_states,
    tsave=tsave,
    params_to_optimize=init_drive_params,
    optimizer=optimizer,
    options=options,
    filepath="tmp_3.h5py"
)
