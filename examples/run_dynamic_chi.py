import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from dynamiqs import Options, grape, timecallable, dag, tensor, basis, destroy, eye, unit
from dynamiqs.utils.file_io import generate_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic chi sim")
    parser.add_argument("--idx", default=-1, type=int, help="idx to scan over")
    parser.add_argument("--gate", default="error_parity", type=str,
                        help="type of gate. Can be error_parity_g, error_parity_plus, ...")
    parser.add_argument("--c_dim", default=3, type=int, help="cavity hilbert dim cutoff")
    parser.add_argument("--t_dim", default=3, type=int, help="tmon hilbert dim cutoff")
    parser.add_argument("--Kerr", default=0.100, type=float, help="transmon Kerr in GHz")
    parser.add_argument("--chi_max", default=0.005, type=float, help="max chi drive in GHz")
    parser.add_argument("--dt", default=10.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=200.0, type=float, help="gate time")
    parser.add_argument("--scale", default=0.001, type=float, help="randomization scale for initial pulse")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=1, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.9995, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=87336259, type=int, help="rng seed for random initial pulses")
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
    H0 = (-2.0 * jnp.pi * args.Kerr * 0.5 * dag(b) @ dag(b) @ b @ b
          -2.0 * jnp.pi * 0.00 * dag(a) @ a @ dag(b) @ b)
    H1 = [dag(a) @ a @ dag(b) @ b, b + dag(b), 1j * (b - dag(b))]
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

    rng = np.random.default_rng(args.rng_seed)
    init_drive_params = 2.0 * jnp.pi * args.scale * rng.random((len(H1), ntimes))

    def H_func(t, drive_params):
        H = H0
        for drive_idx in range(len(H1)):
            if drive_idx == 0:
                total_drive = jnp.clip(
                    envelope * drive_params[drive_idx],
                    a_min=-2.0 * np.pi * args.chi_max,
                    a_max=2.0 * np.pi * args.chi_max,
                )
            else:
                total_drive = envelope * drive_params[drive_idx]
            drive_spline = InterpolatedUnivariateSpline(
                tsave, total_drive, endpoints="natural"
            )
            H = H + drive_spline(t) * H1[drive_idx]
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
            if drive_idx == 0:
                total_drive = jnp.clip(
                    envelope * opt_params[drive_idx],
                    a_min=-2.0 * np.pi * args.chi_max,
                    a_max=2.0 * np.pi * args.chi_max,
                )
            else:
                total_drive = envelope * opt_params[drive_idx]
            drive_spline = InterpolatedUnivariateSpline(
                tsave, total_drive, endpoints="natural"
            )
            plt.plot(finer_times, drive_spline(finer_times)/(2.0 * np.pi), label=f"I_{drive_idx}")
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("pulse amplitude [GHz]")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename[:-5]+"_pulse.pdf")
        plt.show()
