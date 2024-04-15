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
    parser.add_argument("--max_amp", default=[0.001, 0.05, 0.05], help="max drive amp in GHz")
    parser.add_argument("--dt", default=25.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=1000.0, type=float, help="gate time")
    parser.add_argument("--ramp_nts", default=2, type=int, help="numper of points in ramps")
    parser.add_argument("--scale", default=1e-5, type=float, help="randomization scale for initial pulse")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=1, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=300, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.9990, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=46, type=int, help="rng seed for random initial pulses")  # 87336259
    parser.add_argument("--include_low_frequency_noise", default=0, type=int,
                        help="whether to batch over different realizations of low-frequency noise")
    parser.add_argument("--num_freq_shift_trajs", default=5, type=int,
                        help="number of trajectories to sample low-frequency noise for")
    parser.add_argument("--sample_rate", default=1.0, type=float, help="rate at which to sample noise (in us^-1)")
    parser.add_argument("--relative_PSD_strength", default=1e-6, type=float,
                        help="std-dev of frequency shifts given by sqrt(relative_PSD_strength * sample_rate)")
    parser.add_argument("--f0", default=1e-2, type=float, help="cutoff frequency for 1/f noise (in us^-1)")
    parser.add_argument("--white", default=0, type=int, help="white or 1/f noise")
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    parser_args = parser.parse_args()
    if parser_args.idx == -1:
        filename = generate_file_path("h5py", f"dynamic_chi_{parser_args.gate}", "out")
    else:
        filename = f"out/{str(parser_args.idx).zfill(5)}_dynamic_chi_{parser_args.gate}.h5py"
    c_dim = parser_args.c_dim

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
    options = Options(target_fidelity=parser_args.target_fidelity, epochs=parser_args.epochs, coherent=coherent)

    a = tensor(destroy(parser_args.c_dim), eye(parser_args.t_dim))
    b = tensor(eye(parser_args.c_dim), destroy(parser_args.t_dim))
    H0 = -2.0 * jnp.pi * parser_args.Kerr * 0.5 * dag(b) @ dag(b) @ b @ b
    H1 = [dag(a) @ a @ dag(b) @ b, b + dag(b), 1j * (b - dag(b))]
    # H1 = [dag(a) @ a @ dag(b) @ b, ]
    if type(parser_args.max_amp) is float:
        max_amp = len(H1) * [2.0 * jnp.pi * parser_args.max_amp]
    else:
        max_amp = 2.0 * jnp.pi * jnp.asarray(parser_args.max_amp)
    if parser_args.gate == "error_parity_g":
        initial_states = [tensor(basis(parser_args.c_dim, c_idx), basis(parser_args.t_dim, 0))
                          for c_idx in range(2)]
        final_states = [tensor(basis(parser_args.c_dim, c_idx), basis(parser_args.t_dim, c_idx % 2))
                        for c_idx in range(2)]
    elif parser_args.gate == "error_parity_plus":
        initial_states = [tensor(basis(parser_args.c_dim, c_idx), unit(basis(parser_args.t_dim, 0) + basis(parser_args.t_dim, 1)))
                          for c_idx in range(2)]
        final_states = [tensor(basis(parser_args.c_dim, c_idx),
                               unit(basis(parser_args.t_dim, 0) + (-1) ** (c_idx % 2) * basis(parser_args.t_dim, 1)))
                        for c_idx in range(2)]
    else:
        raise RuntimeError("gate type not supported")

    if parser_args.include_low_frequency_noise:

        def obtain_noise_spline(idx):
            noise_t_list, noise_shifts, _, freq_list, psd = generate_noise_trajectory(
                parser_args.sample_rate, parser_args.time, parser_args.relative_PSD_strength,
                parser_args.f0, parser_args.white, parser_args.rng_seed * idx
            )
            noise_spline_coeffs = dx.backward_hermite_coefficients(noise_t_list, noise_shifts)
            return dx.CubicInterpolation(noise_t_list, noise_spline_coeffs)
        additional_drive_args = jnp.arange(parser_args.num_freq_shift_trajs)
        finer_times = jnp.linspace(0.0, parser_args.time, 201)
        fig, ax = plt.subplots()
        for idx in range(parser_args.num_freq_shift_trajs):
            _noise_spline = obtain_noise_spline(idx)
            plt.plot(finer_times, _noise_spline.evaluate(finer_times))
        plt.show()
    else:
        additional_drive_args = None

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

    def H_func(t, drive_params, noise_idx):
        H = H0
        if parser_args.include_low_frequency_noise:
            # extra factor of 2 is because Aniket defines it as 2 pi sigmaz
            noise_spline = obtain_noise_spline(noise_idx)
            H = H + 2.0 * jnp.pi * 2.0 * noise_spline.evaluate(t) * dag(b) @ b
        for drive_idx in range(len(H1)):
            drive_amp = _drive_at_time(t, drive_params[drive_idx], max_amp[drive_idx])
            H = H + drive_amp * H1[drive_idx]
        return H


    H_tc = timecallable(H_func, args=(init_drive_params, 0))

    opt_params = grape(
        H_tc,
        initial_states=initial_states,
        target_states=final_states,
        tsave=tsave,
        params_to_optimize=init_drive_params,
        additional_drive_args=additional_drive_args,
        filepath=filename,
        optimizer=optimizer,
        options=options,
        init_params_to_save=parser_args.__dict__,
    )

    if parser_args.plot:

        finer_times = jnp.linspace(0.0, parser_args.time, 201)
        fig, ax = plt.subplots()
        for drive_idx in range(len(H1)):
            drive_amps = _drive_at_time(
                finer_times, opt_params[drive_idx], max_amp[drive_idx]
            ) / (2.0 * np.pi)
            init_drive_amps = _drive_at_time(
                finer_times, init_drive_params[drive_idx], max_amp[drive_idx]
            ) / (2.0 * np.pi)
            plt.plot(finer_times, drive_amps, label=f"I_{drive_idx}")
            plt.plot(finer_times, init_drive_amps, label=f"I_{drive_idx}_init")
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
            ket = tensor(basis(parser_args.c_dim, c_idx), basis(parser_args.t_dim, t_idx))
            return ket @ dag(ket)

        H_opt = H_tc = timecallable(H_func, args=(opt_params, 0))
        result = sesolve(H_tc, initial_states, finer_times,
                         exp_ops=[Pij(c_idx, t_idx)
                                  for c_idx in range(parser_args.c_dim)
                                  for t_idx in range(parser_args.t_dim)])
        fig, ax = plt.subplots()
        state_idx = 0
        for c_idx in range(parser_args.c_dim):
            for t_idx in range(parser_args.t_dim):
                plt.plot(finer_times, result.expects[0, state_idx],
                         label=f"|{c_idx},{t_idx}>")
                state_idx += 1
        ax.legend()
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("population")
        plt.show()

        fig, ax = plt.subplots()
        state_idx = 0
        for c_idx in range(parser_args.c_dim):
            for t_idx in range(parser_args.t_dim):
                plt.plot(finer_times, result.expects[1, state_idx],
                         label=f"|{c_idx},{t_idx}>")
                state_idx += 1
        ax.legend()
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("population")
        plt.show()
