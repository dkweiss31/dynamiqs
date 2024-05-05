import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from dynamiqs import Options, grape, timecallable, dag, tensor, basis, destroy, eye, unit, mcsolve
from dynamiqs import generate_noise_trajectory
from dynamiqs import sesolve
from dynamiqs.utils.fidelity import all_X_Y_Z_states
from dynamiqs.utils.file_io import generate_file_path
from dynamiqs.time_array import BatchedCallable
import diffrax as dx
from cycler import cycler

color_cycler = plt.rcParams['axes.prop_cycle']
ls_cycler = cycler(ls=['-', '--', '-.', ':'])
alpha_cycler = cycler(alpha=[1.0, 0.6, 0.2])
lw_cycler = cycler(lw=[2.0, 1.0])
color_ls_alpha_cycler = alpha_cycler * lw_cycler * ls_cycler * color_cycler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic chi sim")
    parser.add_argument("--idx", default=-1, type=int, help="idx to scan over")
    parser.add_argument("--gate", default="error_parity_plus_gf", type=str,
                        help="type of gate. Can be error_parity_g, error_parity_plus, ...")
    parser.add_argument("--grape_type", default="jumps", type=str, help="can be unitary, jumps or unitary_and_jumps")
    parser.add_argument("--c_dim", default=4, type=int, help="cavity hilbert dim cutoff")
    parser.add_argument("--t_dim", default=3, type=int, help="tmon hilbert dim cutoff")
    parser.add_argument("--Kerr", default=0.100, type=float, help="transmon Kerr in GHz")
    parser.add_argument("--max_amp", default=[0.001, 0.002, 0.05, 0.05], help="max drive amp in GHz")
    parser.add_argument("--dt", default=20.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=700.0, type=float, help="gate time")
    parser.add_argument("--ramp_nts", default=4, type=int, help="numper of points in ramps")
    parser.add_argument("--scale", default=1e-5, type=float, help="randomization scale for initial pulse")
    parser.add_argument("--learning_rate", default=0.0005, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=0, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=2000, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.990, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=854, type=int, help="rng seed for random initial pulses")  # 87336259
    parser.add_argument("--include_low_frequency_noise", default=0, type=int,
                        help="whether to batch over different realizations of low-frequency noise")
    parser.add_argument("--num_freq_shift_trajs", default=5, type=int,
                        help="number of trajectories to sample low-frequency noise for")
    parser.add_argument("--sample_rate", default=1.0, type=float, help="rate at which to sample noise (in us^-1)")
    parser.add_argument("--relative_PSD_strength", default=1e-6, type=float,
                        help="std-dev of frequency shifts given by sqrt(relative_PSD_strength * sample_rate)")
    parser.add_argument("--f0", default=1e-2, type=float, help="cutoff frequency for 1/f noise (in us^-1)")
    parser.add_argument("--white", default=0, type=int, help="white or 1/f noise")
    parser.add_argument("--T1", default=10000, type=float, help="T1 of the transmon in ns. If not infinity, "
                                                                 "includes jumps")
    parser.add_argument("--ntraj", default=20, type=int, help="number of jump trajectories")
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    parser_args = parser.parse_args()
    if parser_args.idx == -1:
        filename = generate_file_path("h5py", f"dynamic_chi_{parser_args.gate}", "out")
    else:
        filename = f"out/{str(parser_args.idx).zfill(5)}_dynamic_chi_{parser_args.gate}.h5py"
    c_dim = parser_args.c_dim
    t_dim = parser_args.t_dim

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
        ntraj=parser_args.ntraj,
        one_jump_only=True,
    )

    a = tensor(destroy(c_dim), eye(t_dim))
    b = tensor(eye(c_dim), destroy(t_dim))
    g_proj = tensor(eye(c_dim), basis(t_dim, 0) @ dag(basis(t_dim, 0)))
    e_proj = tensor(eye(c_dim), basis(t_dim, 1) @ dag(basis(t_dim, 1)))
    f_proj = tensor(eye(c_dim), basis(t_dim, 2) @ dag(basis(t_dim, 2)))
    gf_proj = tensor(eye(c_dim), basis(t_dim, 0) @ dag(basis(t_dim, 2)))
    ge_proj = tensor(eye(c_dim), basis(t_dim, 0) @ dag(basis(t_dim, 1)))
    ef_proj = tensor(eye(c_dim), basis(t_dim, 1) @ dag(basis(t_dim, 2)))
    # H0 = -2.0 * jnp.pi * parser_args.Kerr * 0.5 * dag(b) @ dag(b) @ b @ b
    H0 = 0.0 * b
    H1 = [dag(a) @ a @ e_proj, dag(a) @ a @ f_proj, gf_proj + dag(gf_proj), 1j * (gf_proj - dag(gf_proj))]
    # H1 = [dag(a) @ a @ e_proj, dag(a) @ a @ f_proj, b + dag(b), 1j * (b - dag(b))]
    # H1 = [dag(a) @ a @ dag(b) @ b, b + dag(b), 1j * (b - dag(b))]
    # H1 = [dag(a) @ a @ dag(b) @ b, ]
    if parser_args.T1 != np.inf:
        jump_ops = [jnp.sqrt(1. / parser_args.T1) * b, ]
    else:
        jump_ops = None
    if type(parser_args.max_amp) is float:
        max_amp = len(H1) * [2.0 * jnp.pi * parser_args.max_amp]
    elif len(parser_args.max_amp) == len(H1):
        max_amp = 2.0 * jnp.pi * jnp.asarray(parser_args.max_amp)
    else:
        raise RuntimeError("max_amp needs to be a float or have the same dimension as H1")
    if parser_args.gate == "error_parity_g":
        initial_states = [tensor(basis(c_dim, c_idx), basis(t_dim, 0))
                          for c_idx in range(2)]
        final_states = [tensor(basis(c_dim, c_idx), basis(t_dim, c_idx % 2))
                        for c_idx in range(2)]
        final_states_traj = None
    elif parser_args.gate == "error_parity_plus":
        initial_states = [tensor(basis(c_dim, c_idx), unit(basis(t_dim, 0) + basis(t_dim, 1)))
                          for c_idx in range(2)]
        final_states = [tensor(basis(c_dim, c_idx),
                               unit(basis(t_dim, 0) + (-1) ** (c_idx % 2) * basis(t_dim, 1)))
                        for c_idx in range(2)]
        final_states_traj = None
    elif parser_args.gate == "error_parity_plus_gf":
        initial_states = [tensor(basis(c_dim, c_idx), unit(basis(t_dim, 0) + basis(t_dim, 2)))
                          for c_idx in range(2)]
        final_states = [tensor(basis(c_dim, c_idx),
                               unit(basis(t_dim, 0) + (-1) ** (c_idx % 2) * basis(t_dim, 2)))
                        for c_idx in range(2)]
        final_states_traj = [
            tensor(basis(c_dim, c_idx), basis(t_dim, 1)) for c_idx in range(2)
        ]
    else:
        raise RuntimeError("gate type not supported")

    if parser_args.coherent == 0:
        initial_states = all_X_Y_Z_states(initial_states)
        final_states = all_X_Y_Z_states(final_states)
        if final_states_traj is not None:
            final_states_traj = all_X_Y_Z_states(final_states_traj)

    if parser_args.include_low_frequency_noise:
        noise_t_list, noise_shifts, _, freq_list, psd = generate_noise_trajectory(
            parser_args.num_freq_shift_trajs, parser_args.sample_rate,
            parser_args.time, parser_args.relative_PSD_strength,
            parser_args.f0, parser_args.white, parser_args.rng_seed
        )
        noise_coeffs = dx.backward_hermite_coefficients(
            noise_t_list, noise_shifts.swapaxes(0, 1)
        )
        noise_spline = dx.CubicInterpolation(noise_t_list, noise_coeffs)
        finer_times = jnp.linspace(0.0, parser_args.time, 201)
        fig, ax = plt.subplots()
        for idx in range(parser_args.num_freq_shift_trajs):
            plt.plot(finer_times, noise_spline.evaluate(finer_times)[idx])
        plt.show()

    rng = np.random.default_rng(parser_args.rng_seed)
    init_drive_params = 2.0 * jnp.pi * (-2.0 * parser_args.scale * rng.random((len(H1), ntimes)) + parser_args.scale)

    def _drives_at_time(t, drive_params):
        drive_w_envelope = jnp.einsum("t,dt->dt", envelope, drive_params)
        total_drive = jnp.clip(
            drive_w_envelope,
            a_min=-max_amp,
            a_max=max_amp,
        )
        drive_coeffs = dx.backward_hermite_coefficients(tsave, total_drive)
        drive_spline = dx.CubicInterpolation(tsave, drive_coeffs)
        return drive_spline.evaluate(t)

    def H_func(t, drive_params):
        drive_amps = _drives_at_time(t, drive_params)
        drive_Hs = jnp.einsum("d,dij->ij", drive_amps, H1)
        H = H0 + drive_Hs
        if parser_args.include_low_frequency_noise:
            # extra factor of 2 is because Aniket defines it as 2 pi sigmaz
            H_freq_shift = jnp.einsum(
                "i,jk->ijk", 2.0 * jnp.pi * 2.0 * noise_spline.evaluate(t), dag(b) @ b
            )
            H = H[None, :, :] + H_freq_shift
        return H

    H_tc = BatchedCallable(H_func)
    # H_tc = timecallable(H_func, args=(init_drive_params, 0))

    opt_params = grape(
        H_tc,
        initial_states=initial_states,
        target_states=final_states,
        tsave=tsave,
        params_to_optimize=init_drive_params,
        grape_type=parser_args.grape_type,
        jump_ops=jump_ops,
        target_states_traj=final_states_traj,
        filepath=filename,
        optimizer=optimizer,
        options=options,
        init_params_to_save=parser_args.__dict__,
    )

    if parser_args.plot:

        finer_times = jnp.linspace(0.0, parser_args.time, 201)
        fig, ax = plt.subplots()
        drive_amps = _drives_at_time(finer_times, opt_params) / (2.0 * np.pi)
        init_drive_amps = _drives_at_time(finer_times, init_drive_params) / (2.0 * np.pi)
        for drive_idx in range(len(H1)):
            plt.plot(finer_times, drive_amps[drive_idx], label=f"I_{drive_idx}")
            plt.plot(finer_times, init_drive_amps[drive_idx], label=f"I_{drive_idx}_init")
        plt.plot(finer_times, (np.pi / (2.0 * np.pi * tsave[-1])) * jnp.ones_like(finer_times),
                 ls="--", color="black", label="chi")
        plt.plot(finer_times, (-np.pi / (2.0 * np.pi * tsave[-1])) * jnp.ones_like(finer_times),
                 ls="--", color="black")
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("pulse amplitude [GHz]")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename[:-5]+"_pulse.pdf")
        plt.show()

        def Pij(c_idx, t_idx):
            ket = tensor(basis(c_dim, c_idx), basis(t_dim, t_idx))
            return ket @ dag(ket)

        H_opt = H_tc = timecallable(H_func, args=(opt_params, 0))
        if parser_args.gate == "error_parity_plus_gf":
            e_idx = 2
        else:
            e_idx = 1

        X_ops = [
            tensor(basis(c_dim, c_idx), basis(t_dim, 0)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)))
            + tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, 0)))
            for c_idx in range(c_dim)
        ]
        X_labels = [f"X_{c_idx}" for c_idx in range(c_dim)]
        Y_ops = [
            1j * tensor(basis(c_dim, c_idx), basis(t_dim, 0)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)))
            - 1j * tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, 0)))
            for c_idx in range(c_dim)
        ]
        Y_labels = [f"Y_{c_idx}" for c_idx in range(c_dim)]
        Z_ops = [
            tensor(basis(c_dim, c_idx), basis(t_dim, 0)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, 0)))
            - tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)))
            for c_idx in range(c_dim)
        ]
        exp_ops = X_ops+Y_ops+Z_ops
        Z_labels = [f"Z_{c_idx}" for c_idx in range(c_dim)]
        labels = X_labels + Y_labels + Z_labels

        if parser_args.T1 != np.inf:
            result = mcsolve(H_tc, jump_ops, initial_states, finer_times, exp_ops=exp_ops)
        else:
            result = sesolve(H_tc, initial_states, finer_times, exp_ops=exp_ops)

        for state_idx in range(len(initial_states)):
            fig, ax = plt.subplots()
            expects = result.expects[state_idx]
            for e_result, label, sty in zip(expects, labels, color_ls_alpha_cycler):
                plt.plot(finer_times, e_result, label=label, **sty)
            ax.legend()
            ax.set_xlabel("time [ns]")
            ax.set_ylabel("population")
            ax.set_title(f"state_idx={state_idx}")
            plt.show()
