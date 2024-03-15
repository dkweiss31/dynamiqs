import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import qutip as qt
import scqubits as scq
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

import dynamiqs as dq
from dynamiqs import Options, grape, timecallable
from dynamiqs._utils import cdtype
from dynamiqs.utils.file_io import generate_file_path
from dynamiqs.utils.fidelity import infidelity_coherent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchQOC sim of second order bin code")
    parser.add_argument("--idx", default=-1, type=int, help="idx to scan over")
    parser.add_argument("--gate", default="prep", type=str,
                        help="type of gate. Can be X_pi, X_piby2, error_recovery, prep, prep_pm")
    parser.add_argument("--c_dim", default=9, type=int, help="hilbert dim cutoff")
    parser.add_argument("--EJ", default=22.0, type=float, help="qubit EJ")
    parser.add_argument("--EC", default=0.15167, type=float, help="qubit EC")
    parser.add_argument("--dt", default=5.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=50.0, type=float, help="gate time")
    parser.add_argument("--scale", default=0.01, type=float, help="randomization scale for initial pulse")
    # parser.add_argument("--bandwidth", default=1.0, type=float, help="pulse bandwidth, fraction of 1/dt/2")
    # parser.add_argument("--ringup", default=1, type=int, help="number of timesteps dedicated to ringup")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=1, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.9995, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=873545436259, type=int, help="rng seed for random initial pulses")
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    # parser.add_argument("--initial_pulse_filepath", default=None,
    #                     type=str, help="initial pulse filepath")
    args = parser.parse_args()
    if args.idx == -1:
        filename = generate_file_path("h5py", f"second_bin_mango_{args.gate}", "out")
    else:
        filename = f"out/{str(args.idx).zfill(5)}_second_bin_mango{args.gate}.h5py"
    c_dim = args.c_dim

    optimizer = optax.adam(learning_rate=args.learning_rate, b1=args.b1, b2=args.b2)
    ntimes = int(args.time // args.dt) + 1
    tsave = jnp.linspace(0, args.time, ntimes)
    # force the control endpoints to be at zero
    envelope = jnp.concatenate((jnp.array([0.0]), jnp.ones(ntimes - 2), jnp.array([0.0])))
    if args.coherent == 0:
        coherent = False
    else:
        coherent = True
    options = Options(target_fidelity=args.target_fidelity, epochs=args.epochs, coherent=coherent)

    tmon = scq.Transmon(EJ=args.EJ, EC=args.EC, ng=0.0, ncut=41, truncated_dim=c_dim)
    hilbert_space = scq.HilbertSpace([tmon])
    hilbert_space.generate_lookup()
    evals = hilbert_space["evals"][0]
    E01 = evals[1] - evals[0]
    _H0_bare = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
    H0_bare = jnp.asarray(_H0_bare, dtype=cdtype())
    rot_frame_mat = 2.0 * np.pi * E01 * np.diag(np.arange(c_dim))
    H0 = jnp.asarray(_H0_bare - qt.Qobj(rot_frame_mat), dtype=cdtype())
    drive_op = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)
    zero_log = (qt.basis(c_dim, 0) + np.sqrt(3) * qt.basis(c_dim, 4)).unit()
    one_log = (np.sqrt(3) * qt.basis(c_dim, 2) + qt.basis(c_dim, 6)).unit()
    E_zero = (np.sqrt(3) * qt.basis(c_dim, 0) - qt.basis(c_dim, 4)).unit()
    E_one = (qt.basis(c_dim, 2) - np.sqrt(3) * qt.basis(c_dim, 6)).unit()
    if args.gate == "error_recovery":
        initial_states = [E_zero, E_one]
        final_states = [zero_log, one_log]
    elif args.gate == "X_pi":
        initial_states = [zero_log, one_log]
        final_states = [one_log, zero_log]
    elif args.gate == "X_piby2":
        initial_states = [zero_log, one_log]
        final_states = [(zero_log + one_log).unit(), (zero_log - one_log).unit()]
    elif args.gate == "prep":
        initial_states = [qt.basis(c_dim, 0), qt.basis(c_dim, 1)]
        final_states = [zero_log, one_log]
    elif args.gate == "prep_pm":
        initial_states = [qt.basis(c_dim, 0), qt.basis(c_dim, 1)]
        final_states = [(zero_log + one_log).unit(), (zero_log - one_log).unit()]
    else:
        raise RuntimeError("gate not one of the supported options")
    # forbidden_states = [
    #     one_log + E_one, zero_log + E_zero, 0.0 * one_log, 0.0 * one_log
    # ]
    eval_diffs = (evals[1:] - evals[0:-1])
    drive_freqs = eval_diffs[0:7]
    H1 = [jnp.asarray(drive_op, dtype=cdtype()), ] * len(drive_freqs)
    rng = np.random.default_rng(args.rng_seed)
    init_drive_params = args.scale * rng.random((2 * len(drive_freqs), ntimes))

    def _I_Q_splines(drive_params, drive_idx):
        I_drive_params = drive_params[2 * drive_idx]
        Q_drive_params = drive_params[2 * drive_idx + 1]
        I_drive_params_env = envelope * I_drive_params
        Q_drive_params_env = envelope * Q_drive_params
        I_spline = InterpolatedUnivariateSpline(tsave, I_drive_params_env, endpoints="natural")
        Q_spline = InterpolatedUnivariateSpline(tsave, Q_drive_params_env, endpoints="natural")
        return I_spline, Q_spline


    def H_func(t, drive_params):
        state_indices = jnp.arange(c_dim)
        rotating_frame_matrix = jnp.exp(
            1j * (jnp.reshape(state_indices, (-1, 1)) - state_indices) * t * 2.0 * np.pi * E01
        )
        H = H0
        for drive_idx in range(len(H1)):
            I_rotating_frame_w_drive = rotating_frame_matrix * jnp.cos(2.0 * np.pi * drive_freqs[drive_idx] * t)
            Q_rotating_frame_w_drive = rotating_frame_matrix * jnp.sin(2.0 * np.pi * drive_freqs[drive_idx] * t)
            I_spline, Q_spline = _I_Q_splines(drive_params, drive_idx)
            H = H + (I_rotating_frame_w_drive * I_spline(t) + Q_rotating_frame_w_drive * Q_spline(t)) * H1[drive_idx]
        return H

    H = timecallable(H_func, args=(init_drive_params,))
    opt_params = grape(
        H,
        initial_states=initial_states,
        target_states=final_states,
        tsave=tsave,
        params_to_optimize=init_drive_params,
        filepath=filename,
        optimizer=optimizer,
        options=options
    )

    if args.plot:

        finer_times = jnp.linspace(0.0, args.time, 201)
        H_ideal = timecallable(H_func, args=(opt_params,))
        res_bare = dq.sesolve(H_ideal, initial_states, finer_times,
                              exp_ops=[dq.basis(c_dim, i) @ dq.tobra(dq.basis(c_dim, i)) for i in range(c_dim)]
                              )
        infid = infidelity_coherent(res_bare.states[..., -1, :, :], jnp.asarray(final_states, dtype=cdtype()))
        print(f"verified fidelity of {1 - infid}")
        fig, ax = plt.subplots()
        for drive_idx in range(len(H1)):
            I_spline, _ = _I_Q_splines(opt_params, drive_idx)
            plt.plot(finer_times, I_spline(finer_times)/(2.0 * np.pi), label=f"I_{drive_idx}")
        plt.gca().set_prop_cycle(None)
        for drive_idx in range(len(H1)):
            _, Q_spline = _I_Q_splines(opt_params, drive_idx)
            plt.plot(finer_times, Q_spline(finer_times)/(2.0 * np.pi), label=f"Q_{drive_idx}", ls="--")
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("pulse amplitude [GHz]")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename[:-5]+"_pulse.pdf")
        plt.show()
        fig, ax = plt.subplots()
        for idx in range(c_dim):
            plt.plot(finer_times, res_bare.expects[0, idx], label=f"{idx}")
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("population")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename[:-5] + "_state_0.pdf")
        plt.show()
        fig, ax = plt.subplots()
        for idx in range(c_dim):
            plt.plot(finer_times, res_bare.expects[1, idx], label=f"{idx}")
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("population")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename[:-5] + "_state_1.pdf")
        plt.show()
