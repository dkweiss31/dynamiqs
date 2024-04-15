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
    parser.add_argument("--c_dim_1", default=4, type=int, help="hilbert dim cutoff 1")
    parser.add_argument("--c_dim_2", default=8, type=int, help="hilbert dim cutoff 2")
    parser.add_argument("--EJ_1", default=12.606, type=float, help="ancilla qubit EJ")
    parser.add_argument("--EJ_2", default=30.0, type=float, help="data qubit EJ")
    parser.add_argument("--EC_1", default=0.270, type=float, help="ancilla qubit EC")
    parser.add_argument("--EC_2", default=0.110, type=float, help="data qubit EC")
    parser.add_argument("--g", default=0.006, type=float, help="qudit-qudit coupling strength")  # 0.00322
    parser.add_argument("--dt", default=100.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=1000.0, type=float, help="gate time")
    parser.add_argument("--ramp_nts", default=2, type=int, help="numper of points in ramps")
    parser.add_argument("--scale", default=0.0001, type=float, help="randomization scale for initial pulse")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=1, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.9995, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=873545436259, type=int, help="rng seed for random initial pulses")
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    parser_args = parser.parse_args()
    if parser_args.idx == -1:
        filename = generate_file_path("h5py", f"second_bin_mango_{parser_args.gate}", "out")
    else:
        filename = f"out/{str(parser_args.idx).zfill(5)}_second_bin_mango{parser_args.gate}.h5py"
    N = parser_args.c_dim_1 * parser_args.c_dim_2

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

    # can swap the roles of the transmons by simply renaming tmon_1 -> tmon_2, tmon_2 -> tmon_1,
    # leaving everything else the same (excpet for c_dims)
    tmon_ancilla = scq.Transmon(EJ=parser_args.EJ_1, EC=parser_args.EC_1, ng=0.0, ncut=41, truncated_dim=parser_args.c_dim_1)
    evals_ancilla = tmon_ancilla.eigenvals(evals_count=parser_args.c_dim_1)
    tmon_data = scq.Transmon(EJ=parser_args.EJ_2, EC=parser_args.EC_2, ng=0.0, ncut=41, truncated_dim=parser_args.c_dim_2)
    evals_data = tmon_data.eigenvals(evals_count=parser_args.c_dim_2)
    hilbert_space = scq.HilbertSpace([tmon_ancilla, tmon_data])
    hilbert_space.add_interaction(
        g=parser_args.g,
        op1=tmon_ancilla.n_operator,
        op2=tmon_data.n_operator,
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
                         for idx_1 in range(parser_args.c_dim_1) for idx_2 in range(parser_args.c_dim_2)])
    drive_op_1 = hilbert_space.op_in_dressed_eigenbasis(tmon_ancilla.n_operator)
    E_diffs = np.abs([
        evals[hilbert_space.dressed_index((1, 1))] - evals[hilbert_space.dressed_index((0, 1))],
        evals[hilbert_space.dressed_index((1, 3))] - evals[hilbert_space.dressed_index((0, 3))],
        evals[hilbert_space.dressed_index((1, 5))] - evals[hilbert_space.dressed_index((0, 5))],
        (evals[hilbert_space.dressed_index((2, 0))] - evals[hilbert_space.dressed_index((0, 0))]) / 2,
        (evals[hilbert_space.dressed_index((2, 2))] - evals[hilbert_space.dressed_index((0, 2))]) / 2,
        (evals[hilbert_space.dressed_index((2, 4))] - evals[hilbert_space.dressed_index((0, 4))]) / 2,
        (evals[hilbert_space.dressed_index((2, 6))] - evals[hilbert_space.dressed_index((0, 6))]) / 2,
    ])
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
    if parser_args.gate == "parity":
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
    rng = np.random.default_rng(parser_args.rng_seed)
    init_drive_params = 2.0 * jnp.pi * (-2.0 * parser_args.scale * rng.random((len(H1), ntimes)) + parser_args.scale)
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
        options=options,
        init_params_to_save=parser_args.__dict__,
    )
