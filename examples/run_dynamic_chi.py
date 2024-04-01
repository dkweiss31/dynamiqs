import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from dynamiqs import Options, grape, timecallable, dag, tensor, basis, destroy, eye, unit
from dynamiqs import generate_noise_trajectory
from dynamiqs import sesolve
from dynamiqs.utils.file_io import generate_file_path
import diffrax as dx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic chi sim")
    parser.add_argument("--idx", default=-1, type=int, help="idx to scan over")
    parser.add_argument("--gate", default="error_parity_plus", type=str,
                        help="type of gate. Can be error_parity_g, error_parity_plus, ...")
    parser.add_argument("--c_dim", default=4, type=int, help="cavity hilbert dim cutoff")
    parser.add_argument("--t_dim", default=4, type=int, help="tmon hilbert dim cutoff")
    parser.add_argument("--Kerr", default=0.100, type=float, help="transmon Kerr in GHz")
    parser.add_argument("--max_amp", default=[0.003, 0.05, 0.05], help="max drive amp in GHz")
    parser.add_argument("--dt", default=40.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=1000.0, type=float, help="gate time")
    parser.add_argument("--scale", default=0.001, type=float, help="randomization scale for initial pulse")
    parser.add_argument("--learning_rate", default=0.0005, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=1, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=10000, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.9995, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=87336259, type=int, help="rng seed for random initial pulses")
    parser.add_argument("--include_low_frequency_noise", default=1, type=int,
                        help="whether to batch over different realizations of low-frequency noise")
    parser.add_argument("--num_freq_shift_trajs", default=10, type=int,
                        help="number of trajectories to sample low-frequency noise for")
    parser.add_argument("--sample_rate", default=1.0, type=float, help="rate at which to sample noise (in us^-1)")
    parser.add_argument("--relative_PSD_strength", default=1e-6, type=float,
                        help="std-dev of frequency shifts given by sqrt(relative_PSD_strength * sample_rate)")
    parser.add_argument("--f0", default=1e-2, type=float, help="cutoff frequency for 1/f noise (in us^-1)")
    parser.add_argument("--white", default=0, type=int, help="white or 1/f noise")
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    args = parser.parse_args()
    if args.idx == -1:
        filename = generate_file_path("h5py", f"dynamic_chi_{args.gate}", "out")
    else:
        filename = f"out/{str(args.idx).zfill(5)}_dynamic_chi_{args.gate}.h5py"
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

    a = tensor(destroy(args.c_dim), eye(args.t_dim))
    b = tensor(eye(args.c_dim), destroy(args.t_dim))
    H0 = -2.0 * jnp.pi * args.Kerr * 0.5 * dag(b) @ dag(b) @ b @ b
    H1 = [dag(a) @ a @ dag(b) @ b, b + dag(b), 1j * (b - dag(b))]
    if type(args.max_amp) is float:
        max_amp = len(H1) * [2.0 * jnp.pi * args.max_amp]
    else:
        max_amp = 2.0 * jnp.pi * jnp.asarray(args.max_amp)
    if args.gate == "error_parity_g":
        initial_states = [tensor(basis(args.c_dim, c_idx), basis(args.t_dim, 0))
                          for c_idx in range(args.c_dim)]
        final_states = [tensor(basis(args.c_dim, c_idx), basis(args.t_dim, c_idx % 2))
                        for c_idx in range(args.c_dim)]
    elif args.gate == "error_parity_plus":
        initial_states = [tensor(basis(args.c_dim, c_idx), unit(basis(args.t_dim, 0) + basis(args.t_dim, 1)))
                          for c_idx in range(2)]
        final_states = [tensor(basis(args.c_dim, c_idx),
                               unit(basis(args.t_dim, 0) + (-1)**(c_idx % 2) * basis(args.t_dim, 1)))
                        for c_idx in range(2)]
    else:
        raise RuntimeError("gate type not supported")

    if args.include_low_frequency_noise:

        def obtain_shifts(idx):
            t_list, shifts, _, freq_list, psd = generate_noise_trajectory(
                args.sample_rate, args.time, args.relative_PSD_strength, args.f0, args.white, 256383 * idx
            )
            return shifts

        noise_shifts = jax.vmap(obtain_shifts)(jnp.arange(args.num_freq_shift_trajs))
        noise_tlist, _, _, _, _ = generate_noise_trajectory(
                args.sample_rate, args.time, args.relative_PSD_strength, args.f0, args.white, 0
            )
        noise_spline_coeffs = dx.backward_hermite_coefficients(noise_tlist, noise_shifts.swapaxes(0, 1))
        noise_spline = dx.CubicInterpolation(noise_tlist, noise_spline_coeffs)


    rng = np.random.default_rng(args.rng_seed)
    init_drive_params = 2.0 * jnp.pi * args.scale * rng.random((len(H1), ntimes))

    def H_func(t, drive_params):
        H = H0
        if args.include_low_frequency_noise:
            # extra factor of 2 is because Aniket defines it as 2 pi sigmaz
            H = H + jnp.einsum(
                "i,jk->ijk", 2.0 * jnp.pi * 2.0 * noise_spline.evaluate(t), dag(b) @ b
            )
        for drive_idx in range(len(H1)):
            total_drive = jnp.clip(
                envelope * drive_params[drive_idx],
                a_min=-max_amp[drive_idx],
                a_max=max_amp[drive_idx],
            )
            drive_coeffs = dx.backward_hermite_coefficients(tsave, total_drive)
            drive_spline = dx.CubicInterpolation(tsave, drive_coeffs)
            H = H + drive_spline.evaluate(t) * H1[drive_idx]
        return H

    H_tc = timecallable(H_func, args=(init_drive_params,))
    opt_params = grape(
        H_tc,
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
        fig, ax = plt.subplots()
        for drive_idx in range(len(H1)):
            total_drive = jnp.clip(
                envelope * opt_params[drive_idx],
                a_min=-max_amp[drive_idx],
                a_max=max_amp[drive_idx],
            )
            drive_coeffs = dx.backward_hermite_coefficients(tsave, total_drive)
            drive_spline = dx.CubicInterpolation(tsave, drive_coeffs)
            plt.plot(finer_times, drive_spline.evaluate(finer_times)/(2.0 * np.pi), label=f"I_{drive_idx}")
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
            ket = tensor(basis(args.c_dim, c_idx), basis(args.t_dim, t_idx))
            return ket @ dag(ket)

        H_opt = H_tc = timecallable(H_func, args=(opt_params,))
        result = sesolve(H_tc, initial_states, finer_times,
                         exp_ops=[Pij(c_idx, t_idx)
                                  for c_idx in range(args.c_dim)
                                  for t_idx in range(args.t_dim)])
        fig, ax = plt.subplots()
        state_idx = 0
        for c_idx in range(args.c_dim):
            for t_idx in range(args.t_dim):
                plt.plot(finer_times, result.expects[0, state_idx],
                         label=f"|{c_idx},{t_idx}>")
                state_idx += 1
        ax.legend()
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("population")
        plt.show()

        fig, ax = plt.subplots()
        state_idx = 0
        for c_idx in range(args.c_dim):
            for t_idx in range(args.t_dim):
                plt.plot(finer_times, result.expects[1, state_idx],
                         label=f"|{c_idx},{t_idx}>")
                state_idx += 1
        ax.legend()
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("population")
        plt.show()
