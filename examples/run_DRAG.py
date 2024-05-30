import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from dynamiqs import Options, grape, timecallable, dag, basis, destroy
from dynamiqs import sesolve
from dynamiqs.utils.fidelity import all_X_Y_Z_states
from dynamiqs.utils.file_io import generate_file_path
import diffrax as dx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAPE sim")
    parser.add_argument("--idx", default=-1, type=int, help="idx to scan over")
    parser.add_argument("--dim", default=4, type=int, help="tmon hilbert dim cutoff")
    parser.add_argument("--Kerr", default=0.100, type=float, help="transmon Kerr in GHz")
    parser.add_argument("--max_amp", default=[0.1, 0.1], help="max drive amp in GHz")
    parser.add_argument("--dt", default=2.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=20.0, type=float, help="gate time")
    parser.add_argument("--ramp_nts", default=3, type=int, help="numper of points in ramps")
    parser.add_argument("--scale", default=1e-5, type=float, help="randomization scale for initial pulse")
    parser.add_argument("--learning_rate", default=0.0005, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=0, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=2000, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.9995, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=854, type=int, help="rng seed for random initial pulses")  # 87336259
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    parser_args = parser.parse_args()
    if parser_args.idx == -1:
        filename = generate_file_path("h5py", f"DRAG", "out")
    else:
        filename = f"out/{str(parser_args.idx).zfill(5)}_DRAG.h5py"

    dim = parser_args.dim

    optimizer = optax.adam(learning_rate=parser_args.learning_rate, b1=parser_args.b1, b2=parser_args.b2)
    ntimes = int(parser_args.time // parser_args.dt) + 1
    tsave = jnp.linspace(0, parser_args.time, ntimes)
    # force the control endpoints to be at zero
    begin_ramp = (1 - jnp.cos(jnp.linspace(0.0, jnp.pi, parser_args.ramp_nts))) / 2
    envelope = jnp.concatenate(
        (begin_ramp, jnp.ones(ntimes - 2 * parser_args.ramp_nts), jnp.flip(begin_ramp))
    )
    if parser_args.coherent == 0:
        coherent = False
    else:
        coherent = True
    options = Options(
        save_states=False,
        target_fidelity=parser_args.target_fidelity,
        epochs=parser_args.epochs,
        coherent=coherent,
    )
    a = destroy(dim)
    H0 = -0.5 * parser_args.Kerr * 2.0 * jnp.pi * dag(a) @ dag(a) @ a @ a
    H1 = [a + dag(a), 1j * (a - dag(a))]

    if type(parser_args.max_amp) is float:
        max_amp = len(H1) * [2.0 * jnp.pi * parser_args.max_amp]
    elif len(parser_args.max_amp) == len(H1):
        max_amp = 2.0 * jnp.pi * jnp.asarray(parser_args.max_amp)
    else:
        raise RuntimeError("max_amp needs to be a float or have the same dimension as H1")

    initial_states = [basis(dim, 0), basis(dim, 1)]
    final_states = [basis(dim, 1), basis(dim, 0)]

    # need to form superpositions so that the phase information is correct
    if parser_args.coherent == 0:
        initial_states = all_X_Y_Z_states(initial_states)
        final_states = all_X_Y_Z_states(final_states)

    rng = np.random.default_rng(parser_args.rng_seed)
    init_drive_params = 2.0 * jnp.pi * (-2.0 * parser_args.scale * rng.random((len(H1), ntimes)) + parser_args.scale)

    def _drive_at_time(t, drive_param, max_amp):
        total_drive = jnp.clip(
            envelope * drive_param,
            a_min=-max_amp,
            a_max=max_amp,
        )
        drive_coeffs = dx.backward_hermite_coefficients(tsave, total_drive)
        drive_spline = dx.CubicInterpolation(tsave, drive_coeffs)
        return drive_spline.evaluate(t)

    def H_func(t, drive_params):
        H = H0
        for drive_idx in range(len(H1)):
            drive_amp = _drive_at_time(t, drive_params[drive_idx], max_amp[drive_idx])
            H = H + drive_amp * H1[drive_idx]
        return H


    # H_tc = timecallable(H_func, args=(init_drive_params, 0))
    H_tc = jax.tree_util.Partial(H_func)

    opt_params = grape(
        H_tc,
        initial_states=initial_states,
        target_states=final_states,
        tsave=tsave,
        params_to_optimize=init_drive_params,
        filepath=filename,
        optimizer=optimizer,
        options=options,
        init_params_to_save=parser_args.__dict__,
    )

    if parser_args.plot:

        finer_times = jnp.linspace(0.0, parser_args.time, 201)
        fig, ax = plt.subplots()
        drive_amps_X = _drive_at_time(
            finer_times, opt_params[0], max_amp[0]
        ) / (2.0 * np.pi)
        drive_amps_Y = _drive_at_time(
            finer_times, opt_params[1], max_amp[1]
        ) / (2.0 * np.pi)
        plt.plot(finer_times, drive_amps_X, label=f"X")
        plt.plot(finer_times, drive_amps_Y, label=f"Y")
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("pulse amplitude [GHz]")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename[:-5]+"_pulse.pdf")
        plt.show()


        H_opt = H_tc = timecallable(H_func, args=(opt_params, 0))
        result = sesolve(H_tc, initial_states, finer_times,
                         exp_ops=[basis(dim, idx) @ dag(basis(dim, idx)) for idx in range(dim)])
        init_labels = [r"$|0\rangle$", r"$|0\rangle+|1\rangle$", r"$|0\rangle+i|1\rangle$", r"$|1\rangle$"]
        exp_labels = [r"$|0\rangle$", r"$|1\rangle$", r"$|2\rangle$", r"$|3\rangle$"]

        for state_idx in range(len(initial_states)):
            fig, ax = plt.subplots()
            expects = result.expects[state_idx]
            for e_result, label in zip(expects, exp_labels):
                plt.plot(finer_times, e_result, label=label)
            ax.legend()
            ax.set_xlabel("time [ns]")
            ax.set_ylabel("population")
            ax.set_title(f"initial state={init_labels[state_idx]}")
            plt.show()
