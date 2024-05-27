import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import diffrax as dx
import scqubits as scq
from cycler import cycler

import dynamiqs as dq
from dynamiqs import Options, grape, timecallable, basis, unit
from dynamiqs.utils.fidelity import all_X_Y_Z_states
from dynamiqs._utils import cdtype
from dynamiqs.utils.file_io import generate_file_path

color_cycler = plt.rcParams['axes.prop_cycle']
ls_cycler = cycler(ls=['-', '--', '-.', ':'])
alpha_cycler = cycler(alpha=[1.0, 0.6, 0.2])
lw_cycler = cycler(lw=[2.0, 1.0])
color_ls_alpha_cycler = alpha_cycler * lw_cycler * ls_cycler * color_cycler


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
    parser.add_argument("--max_amp", default=0.01, help="max drive amp in GHz")
    parser.add_argument("--dt", default=20.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=500.0, type=float, help="gate time")
    parser.add_argument("--ramp_nts", default=2, type=int, help="numper of points in ramps")
    parser.add_argument("--scale", default=0.0001, type=float, help="randomization scale for initial pulse")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=0, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.9995, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=87356259, type=int, help="rng seed for random initial pulses")
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    parser_args = parser.parse_args()
    if parser_args.idx == -1:
        filename = generate_file_path("h5py", f"second_bin_mango_{parser_args.gate}", "out")
    else:
        filename = f"out/{str(parser_args.idx).zfill(5)}_second_bin_mango_{parser_args.gate}.h5py"
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
    options = Options(
        save_states=False,
        target_fidelity=parser_args.target_fidelity,
        epochs=parser_args.epochs,
        coherent=coherent,
        # ntraj=parser_args.ntraj,
        one_jump_only=True,
    )

    tmon_ancilla = scq.Transmon(
        EJ=parser_args.EJ_1, EC=parser_args.EC_1, ng=0.0, ncut=41, truncated_dim=parser_args.c_dim_1
    )
    evals_ancilla = tmon_ancilla.eigenvals(evals_count=parser_args.c_dim_1)
    tmon_data = scq.Transmon(
        EJ=parser_args.EJ_2, EC=parser_args.EC_2, ng=0.0, ncut=41, truncated_dim=parser_args.c_dim_2
    )
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
    drive_op_ancilla = hilbert_space.op_in_dressed_eigenbasis(tmon_ancilla.n_operator)
    drive_op_data = hilbert_space.op_in_dressed_eigenbasis(tmon_data.n_operator)
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
            zero_log, one_log,
            E_zero, E_one,
            E_two, E_three, E_four
        ]
        final_states = [
            zero_log, one_log,
            E_zero_fin, E_one_fin,
            E_two_fin, E_three_fin, E_four_fin
        ]
    elif parser_args.gate == "dephasing_parity":
        E_zero_fin = unit(jnp.sqrt(3.0) * dressed_kets["(1,0)"] - dressed_kets["(1,4)"])
        E_one_fin = unit(dressed_kets["(1,2)"] - jnp.sqrt(3.0) * dressed_kets["(1,6)"])
        initial_states = [zero_log, one_log, E_zero, E_one]
        final_states = [zero_log, one_log, E_zero_fin, E_one_fin]
    else:
        raise RuntimeError("gate type not supported")

    # don't drive the highest included state, included only for convergence purposes
    E_diffs_ancilla = np.diff(evals_ancilla)[0:-1]
    E_diffs_data = np.diff(evals_data)[0:-1]
    E_diffs = np.concatenate((E_diffs_ancilla, E_diffs_data))
    H1 = [jnp.asarray(drive_op_ancilla, dtype=cdtype()), ] * len(E_diffs_ancilla)
    H1 += [jnp.asarray(drive_op_data, dtype=cdtype()), ] * len(E_diffs_data)

    if type(parser_args.max_amp) is float:
        max_amp = len(H1) * jnp.asarray([2.0 * jnp.pi * parser_args.max_amp])
    elif len(parser_args.max_amp) == len(H1):
        max_amp = 2.0 * jnp.pi * jnp.asarray(parser_args.max_amp)
    else:
        raise RuntimeError("max_amp needs to be a float or have the same dimension as H1")

    if parser_args.coherent == 0:
        initial_states = all_X_Y_Z_states(initial_states)
        final_states = all_X_Y_Z_states(final_states)

    rng = np.random.default_rng(parser_args.rng_seed)
    init_drive_params = 2.0 * jnp.pi * (-2.0 * parser_args.scale * rng.random((len(H1), ntimes)) + parser_args.scale)
    rot_frame_drive = jnp.reshape(rot_frame_diag, (-1, 1)) - rot_frame_diag

    def _drive_spline(drive_params):
        # note swap of axes so that time axis is first
        drive_w_envelope = jnp.einsum("t,dt->td", envelope, drive_params)
        total_drive = jnp.clip(
            drive_w_envelope,
            a_min=-max_amp[None, :],
            a_max=max_amp[None, :],
        )
        drive_coeffs = dx.backward_hermite_coefficients(tsave, total_drive)
        drive_spline = dx.CubicInterpolation(tsave, drive_coeffs)
        return drive_spline


    def H_func(t, drive_params):
        drive_amps = (
            jnp.cos(2.0 * np.pi * E_diffs * t)
            * _drive_spline(drive_params).evaluate(t)
        )
        H1_rot = jnp.einsum(
            "ij,dij->dij",
            jnp.exp(1j * rot_frame_drive * t * 2.0 * jnp.pi),
            H1
        )
        drive_Hs = jnp.einsum("d,dij->ij", drive_amps, H1_rot)
        return H0 + drive_Hs

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

    def H_for_plotting(t):
        drive_amps = (
            jnp.cos(2.0 * np.pi * E_diffs * t)
            * _drive_spline(opt_params).evaluate(t)
        )
        drive_Hs = jnp.einsum("d,dij->ij", drive_amps, H1)
        return H0_bare + drive_Hs


    finer_times = np.linspace(0.0, parser_args.time, 401)
    fig, ax = plt.subplots()
    drive_spline_opt = _drive_spline(opt_params)
    for drive_idx, drive in enumerate(opt_params):
        plt.plot(
            finer_times,
            10 ** 3 * drive_spline_opt.evaluate(finer_times)[drive_idx] / (2.0 * np.pi),
        )
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("pulse amplitude [MHz]")
    plt.show()

    H_plot_tc = timecallable(H_for_plotting)
    plot_result = dq.sesolve(H_plot_tc, initial_states, tsave,
                             exp_ops=[dressed_ket(idx_1, idx_2) @ dq.dag(dressed_ket(idx_1, idx_2))
                                      for idx_1 in range(parser_args.c_dim_1)
                                      for idx_2 in range(parser_args.c_dim_2)])

    for state_idx in range(len(initial_states)):
        expects = plot_result.expects[state_idx]

        fig, ax = plt.subplots()
        labels = [f"({idx_1}, {idx_2})" for idx_1 in range(parser_args.c_dim_1)
                  for idx_2 in range(parser_args.c_dim_2)]
        for e_result, label, sty in zip(expects, labels, color_ls_alpha_cycler):
            plt.plot(tsave, e_result, label=label, **sty)
        ax.legend(fontsize=12, ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title(f"state index {state_idx}")
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("population")
        plt.show()
