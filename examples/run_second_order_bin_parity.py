import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import qutip as qt
import diffrax as dx
import scqubits as scq

import dynamiqs as dq
from dynamiqs import Options, grape, timecallable, basis, unit
from dynamiqs._utils import cdtype
from dynamiqs.utils.file_io import generate_file_path
from dynamiqs.utils.fidelity import infidelity_coherent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchQOC sim of second order bin code")
    parser.add_argument("--idx", default=-1, type=int, help="idx to scan over")
    parser.add_argument("--gate", default="parity", type=str,
                        help="type of gate. Can be parity")
    parser.add_argument("--c_dim_1", default=3, type=int, help="hilbert dim cutoff 1")
    parser.add_argument("--c_dim_2", default=7, type=int, help="hilbert dim cutoff 2")
    parser.add_argument("--EJ_1", default=21.22130, type=float, help="qubit 1 EJ")
    parser.add_argument("--EJ_2", default=22.01162, type=float, help="qubit 2 EJ")
    parser.add_argument("--EC_1", default=0.15167, type=float, help="qubit 1 EC")
    parser.add_argument("--EC_2", default=0.15185, type=float, help="qubit 2 EC")
    parser.add_argument("--g", default=0.006, type=float, help="qudit-qudit coupling strength")  # 0.00322
    parser.add_argument("--dt", default=100.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=2000.0, type=float, help="gate time")
    parser.add_argument("--scale", default=0.0001, type=float, help="randomization scale for initial pulse")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=1, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.9995, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=873545436259, type=int, help="rng seed for random initial pulses")
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    args = parser.parse_args()
    if args.idx == -1:
        filename = generate_file_path("h5py", f"second_bin_mango_{args.gate}", "out")
    else:
        filename = f"out/{str(args.idx).zfill(5)}_second_bin_mango{args.gate}.h5py"
    N = args.c_dim_1 * args.c_dim_2

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

    # can swap the roles of the transmons by simply renaming tmon_1 -> tmon_2, tmon_2 -> tmon_1,
    # leaving everything else the same (excpet for c_dims)
    tmon_1 = scq.Transmon(EJ=args.EJ_1, EC=args.EC_1, ng=0.0, ncut=41, truncated_dim=args.c_dim_1)
    evals_1 = tmon_1.eigenvals(evals_count=args.c_dim_1)
    tmon_2 = scq.Transmon(EJ=args.EJ_2, EC=args.EC_2, ng=0.0, ncut=41, truncated_dim=args.c_dim_2)
    evals_2 = tmon_2.eigenvals(evals_count=args.c_dim_2)
    hilbert_space = scq.HilbertSpace([tmon_1, tmon_2])
    hilbert_space.add_interaction(
        g=args.g,
        op1=tmon_1.n_operator,
        op2=tmon_2.n_operator,
        add_hc=False,
    )
    hilbert_space.generate_lookup()
    evals = hilbert_space["evals"][0]
    E01_1 = evals[hilbert_space.dressed_index((1, 0))] - evals[0]
    E01_2 = evals[hilbert_space.dressed_index((0, 1))] - evals[0]
    dressed_idxs = np.arange(N)
    bare_labels = [hilbert_space.bare_index(dressed_idx) for dressed_idx in dressed_idxs]
    rot_frame_diag = jnp.asarray([E01_1 * bare_1 + E01_2 * bare_2
                                  for (bare_1, bare_2) in bare_labels])
    rot_frame_mat = 2.0 * jnp.pi * jnp.diag(rot_frame_diag)
    H0_bare = 2.0 * jnp.pi * jnp.diag(evals - evals[0])
    H0 = H0_bare - rot_frame_mat

    def dressed_ket(bare_label_1, bare_label_2):
        dressed_idx = int(hilbert_space.dressed_index((bare_label_1, bare_label_2)))
        return basis(N, dressed_idx)

    dressed_kets = dict([(f"({idx_1},{idx_2})", dressed_ket(idx_1, idx_2))
                        for idx_1 in range(args.c_dim_1) for idx_2 in range(args.c_dim_2)])
    drive_op_1 = hilbert_space.op_in_dressed_eigenbasis(tmon_1.n_operator)
    drive_op_2 = hilbert_space.op_in_dressed_eigenbasis(tmon_2.n_operator)
    large_matelem_idxs = jnp.argwhere(jnp.abs(drive_op_1.data.toarray()) > 0.5)
    large_matelem_idxs_top = jnp.array([[idx_1, idx_2] for (idx_1, idx_2) in large_matelem_idxs if idx_1 > idx_2])
    E_diffs = evals[large_matelem_idxs_top[:, 0]] - evals[large_matelem_idxs_top[:, 1]]
    zero_log = unit(dressed_kets["(0,0)"] + jnp.sqrt(3.0) * dressed_kets["(0,4)"])
    one_log = unit(jnp.sqrt(3.0) * dressed_kets["(0,2)"] + dressed_kets["(0,6)"])
    E_zero = unit(jnp.sqrt(3.0) * dressed_kets["(0,0)"] - dressed_kets["(0,4)"])
    E_one = unit(dressed_kets["(0,2)"] - jnp.sqrt(3.0) * dressed_kets["(0,6)"])
    E_zero_fin = unit(jnp.sqrt(3.0) * dressed_kets["(2,0)"] - dressed_kets["(2,4)"])
    E_one_fin = unit(dressed_kets["(2,2)"] - jnp.sqrt(3.0) * dressed_kets["(2,6)"])
    E_two = dressed_kets["(0,1)"]
    E_three = dressed_kets["(0,3)"]
    E_four = dressed_kets["(0,5)"]
    E_two_fin = dressed_kets["(1,1)"]
    E_three_fin = dressed_kets["(1,3)"]
    E_four_fin = dressed_kets["(1,5)"]
    if args.gate == "parity":
        initial_states = [
            zero_log, one_log, unit(zero_log + one_log), unit(zero_log + 1j * one_log),
            E_zero, E_one, unit(E_zero + E_one), unit(E_zero + 1j * E_one),
            E_two, E_three, E_four
        ]
        final_states = [
            zero_log, one_log, unit(zero_log + one_log), unit(zero_log + 1j * one_log),
            E_zero_fin, E_one_fin, unit(E_zero_fin + E_one_fin), unit(E_zero_fin + 1j * E_one_fin),
            E_two_fin, E_three_fin, E_four_fin
        ]
    H1 = [jnp.asarray(drive_op_1, dtype=cdtype()), ] * len(E_diffs)
    rng = np.random.default_rng(args.rng_seed)
    init_drive_params = 2.0 * jnp.pi * args.scale * rng.random((2 * len(E_diffs), ntimes))
    rwa_cutoff = jnp.inf
    rot_frame_drive = jnp.reshape(rot_frame_diag, (-1, 1)) - rot_frame_diag

    def H_func(t, drive_params, additional_args):
        H = H0
        # TODO this isn't exactly right since we also need to include the drive time dependence
        rot_frame_drive_rwa = jnp.where(
            np.abs(rot_frame_drive) < rwa_cutoff,
            jnp.exp(1j * rot_frame_drive * t * 2.0 * jnp.pi),
            0.0,
        )
        for drive_idx in range(len(H1) // 2):
            I_drive_coeffs = dx.backward_hermite_coefficients(tsave, envelope * drive_params[2 * drive_idx])
            I_drive_spline = dx.CubicInterpolation(tsave, I_drive_coeffs)
            Q_drive_coeffs = dx.backward_hermite_coefficients(tsave, envelope * drive_params[2 * drive_idx + 1])
            Q_drive_spline = dx.CubicInterpolation(tsave, Q_drive_coeffs)
            H = H + (jnp.cos(2.0 * np.pi * E_diffs[drive_idx] * t)
                     * I_drive_spline.evaluate(t)
                     + jnp.sin(2.0 * np.pi * E_diffs[drive_idx] * t)
                     * Q_drive_spline.evaluate(t)) * rot_frame_drive_rwa * H1[drive_idx]
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
