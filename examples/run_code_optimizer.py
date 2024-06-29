import argparse
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from dynamiqs import Options, grape, timecallable, dag, tensor, basis, destroy, eye, unit, mcsolve, code_optimizer
from dynamiqs import generate_noise_trajectory
from dynamiqs import sesolve
from dynamiqs.utils.fidelity import all_X_Y_Z_states, infidelity_incoherent
from dynamiqs.utils.file_io import generate_file_path, extract_info_from_h5
import diffrax as dx
from cycler import cycler

color_cycler = plt.rcParams['axes.prop_cycle']
ls_cycler = cycler(ls=['-', '--', '-.', ':'])
alpha_cycler = cycler(alpha=[1.0, 0.6, 0.2])
lw_cycler = cycler(lw=[2.0, 1.0])
color_ls_alpha_cycler = alpha_cycler * lw_cycler * ls_cycler * color_cycler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="codeword optimizer binomial")
    parser.add_argument("--idx", default=-1, type=int, help="idx to scan over")
    parser.add_argument("--c_dim", default=7, type=int, help="cavity hilbert dim cutoff")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.9, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.9, type=float, help="decay of learning rate second moment")
    parser.add_argument("--epochs", default=40000, type=int, help="number of epochs")
    parser.add_argument("--target_infidelity", default=0.1, type=float, help="target infidelity")
    parser.add_argument("--kappa_dt", default=0.01, type=float, help="kappa_dt product")
    # parser.add_argument("--n_save", default=100, type=int, help="how often to save")
    parser.add_argument("--rng_seed", default=63055734845647, type=int, help="rng seed for random initial pulses")
    parser_args = parser.parse_args()
    if parser_args.idx == -1:
        filename = generate_file_path("h5py", f"codeword_optimizer", "out")
    else:
        filename = f"out/{str(parser_args.idx).zfill(5)}_codeword_optimizer.h5py"
    c_dim = parser_args.c_dim

    optimizer = optax.adam(learning_rate=parser_args.learning_rate, b1=parser_args.b1, b2=parser_args.b2)
    a = destroy(c_dim)
    jump_ops = [eye(c_dim), a, dag(a) @ a]
    op_to_minimize = jnp.sqrt(parser_args.kappa_dt ** 2 / 2) * a @ a
    # jump_ops = [basis(c_dim, idx) @ dag(basis(c_dim, idx)) for idx in range(c_dim)]
    # op_to_minimize = 0.0 * basis(c_dim, 0) @ dag(basis(c_dim, 0))

    rng = np.random.default_rng(parser_args.rng_seed)
    init_codewords = unit(rng.random((2, c_dim, 1)))

    _ = code_optimizer(
        codewords=init_codewords,
        jump_ops=jump_ops,
        op_to_minimize=op_to_minimize,
        epochs=parser_args.epochs,
        target_infidelity=parser_args.target_infidelity,
        filepath=filename,
        optimizer=optimizer,
        init_params_to_save=parser_args.__dict__,
    )
    data_dict, param_dict = extract_info_from_h5(filepath=filename)
    opt_codewords_idx = np.argmin(data_dict["infidelities"])
    opt_codewords = data_dict["opt_params"][opt_codewords_idx]
    print(opt_codewords[0], opt_codewords[1])
    print(opt_codewords_idx, data_dict["infidelities"][opt_codewords_idx])
    qec_mat = jnp.einsum(
        "wid,lij,kjm,smd->lkws",
        jnp.conj(opt_codewords),
        dag(jump_ops),
        jump_ops,
        opt_codewords
    )
