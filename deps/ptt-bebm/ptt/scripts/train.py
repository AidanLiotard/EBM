import argparse

import h5py
import torch
from rbms.dataset import load_dataset
from rbms.dataset.parser import add_args_dataset
from rbms.map_model import map_model
from rbms.optim import setup_optim
from rbms.parser import (
    add_args_init_rbm,
    add_args_pytorch,
    add_args_saves,
    add_args_train,
    add_grad_args,
    add_sampling_args,
    default_args,
    match_args_dtype,
    remove_argument,
    set_args_default,
)
from rbms.pre_grad import build_pre_grad_update
from rbms.training.implement import _restore_training
from rbms.training.pcd import train
from rbms.training.utils import get_checkpoints
from rbms.utils import get_flagged_updates, get_saved_updates
from ptt.training.utils import get_sampler_kernel_args_ptt

from ptt.generic.classes import PTT, AcceptanceRateException
from ptt.training.utils import init_training_ptt, reset_training

default_args_ptt = {
    "num_swaps": 1,
    "increment": 1,
    "reservoir_size": None,
    "max_n_model": 2,
    "target_n_model": 2,
    "num_steps_annealing": 100,
    "n_sample_steps": 10,
    "target_acc_rate": 0.25,
    "patience": 3,
}


def clean_old_checkpoints(filename, keep_last=5):
    try:
        saved_updates = get_saved_updates(filename)
        ptt_updates = get_flagged_updates(filename, "ptt")

        if len(saved_updates) <= keep_last:
            return

        with h5py.File(filename, "a") as f:
            for upd in saved_updates[:-keep_last]:
                # Preserve initialization (update 1) and PTT ladder structural updates
                if upd != 1 and upd not in ptt_updates:
                    if f"update_{upd}" in f.keys():
                        del f[f"update_{upd}"]
    except Exception:
        # Prevent crash if file is temporarily locked by another process
        pass


def add_args_ptt(parser: argparse.ArgumentParser):
    ptt_args = parser.add_argument_group("PTT")
    ptt_args.add_argument(
        "--num_swaps",
        default=None,
        type=int,
        help="(Defaults to 1). The number of swap steps between two gradient updates.",
    )
    ptt_args.add_argument(
        "--increment",
        default=None,
        type=int,
        help="(Defaults to 1). The number of sampling steps before proposing a swap in PTT.",
    )
    ptt_args.add_argument(
        "--reservoir_size",
        default=None,
        type=int,
        help="(Defaults to 10*num_chains). Size of the reservoir to propose samples.",
    )
    ptt_args.add_argument(
        "--max_n_model",
        default=None,
        type=int,
        help="(Defaults to 2). Maximum number of model before recomputing Reservoir.",
    )
    ptt_args.add_argument(
        "--target_n_model",
        default=None,
        type=int,
        help="(Defaults to 2). Minimum number of models for the Reservoir.",
    )
    ptt_args.add_argument(
        "--num_steps_annealing",
        type=int,
        default=None,
        help="(Defaults to 100). Number of sampling steps between two temperature update when initializing chains with annealing in PTT.",
    )
    ptt_args.add_argument(
        "--n_sample_steps",
        type=int,
        default=None,
        help="(Defaults to 10). The number of sampling steps performed when updating the chain of replicas.",
    )
    ptt_args.add_argument(
        "--target_acc_rate",
        type=float,
        default=None,
        help="(Defaults to 0.25). Target acceptance rate between replicas of PTT.",
    )
    ptt_args.add_argument(
        "--full_sampler",
        action="store_true",
        default=False,
        help="Use all PTT model to generate reservoirs during training.",
    )
    ptt_args.add_argument(
        "--patience",
        type=int,
        default=None,
        help="(Defaults to 3). Patience for early stopping.",
    )
    return parser


def create_parser():
    parser = argparse.ArgumentParser(description="Train a Restricted Boltzmann Machine")
    parser = add_args_dataset(parser)
    parser = add_args_init_rbm(parser)
    parser = add_args_train(parser)
    parser = add_args_ptt(parser)
    parser = add_sampling_args(parser)
    parser = add_grad_args(parser)
    parser = add_args_saves(parser)
    parser = add_args_pytorch(parser)
    remove_argument(parser, "use_torch")
    return parser


def load_args_from_filename(args: dict):
    with h5py.File(args["filename"], "r") as f:
        if args["num_swaps"] is None:
            args["num_swaps"] = f["sampling_args"]["num_swaps"][()].item()
        # if args["beta"] is None:
        #     args["beta"] = f["sampling_args"]["beta"][()].item()
        if args["optim"] is None:
            args["optim"] = str(f["train_args"]["optim"][()])
        if args["batch_size"] is None:
            args["batch_size"] = f["train_args"]["batch_size"][()].item()
        if args["training_type"] is None:
            args["training_type"] = str(f["train_args"]["training_type"][()])
        args["no_center"] = f["grad_args"]["no_center"][()].item()
        args["seed"] = f["dataset_args"]["seed"][()].item()
        args["train_size"] = f["dataset_args"]["train_size"][()].item()
        args["test_size"] = f["dataset_args"]["test_size"][()].item()
        if args["L1"] is None:
            args["L1"] = f["grad_args"]["L1"][()].item()
        if args["L2"] is None:
            args["L2"] = f["grad_args"]["L2"][()].item()
        if args["normalize_grad"] is None:
            args["normalize_grad"] = f["grad_args"]["normalize_grad"][()].item()
        if args["max_norm_grad"] is None:
            args["max_norm_grad"] = f["grad_args"]["max_norm_grad"][()].item()
        if args["data_noise_std"] is None and "data_noise_std" in f["train_args"]:
            args["data_noise_std"] = f["train_args"]["data_noise_std"][()].item()
        if args["data_std"] is None and "data_std" in f["train_args"]:
            args["data_std"] = f["train_args"]["data_std"][()].item()

    return args


def process_args_ptt(args: dict):
    args_torch = {"device": args["device"], "dtype": args["dtype"]}
    args_dataset = {
        "dataset_name": args["dataset"],
        "test_dataset_name": args["test_dataset"],
        "train_size": args["train_size"],
        "test_size": args["test_size"],
        "subset_labels": args["subset_labels"],
        "use_weights": args["use_weights"],
        "alphabet": args["alphabet"],
        "remove_duplicates": args["remove_duplicates"],
        "seed": args["seed"],
    }
    args_grad = {
        "no_center": args["no_center"],
        "normalize_grad": args["normalize_grad"],
        "max_norm_grad": args["max_norm_grad"],
        "L1": args["L1"],
        "L2": args["L2"],
    }
    args_ptt = {
        "num_swaps": args["num_swaps"],
        "increment": args["increment"],
        "reservoir_size": args["reservoir_size"]
        if args["reservoir_size"] is not None
        else 10 * args["num_chains"],
        "max_n_model": args["max_n_model"],
        "target_n_model": args["target_n_model"],
        "num_steps_annealing": args["num_steps_annealing"],
        "n_sample_steps": args["n_sample_steps"],
        "target_acc_rate": args["target_acc_rate"],
        "full_sampler": args["full_sampler"],
        "patience": args["patience"],
    }

    args_train = {
        "optim": args["optim"],
        "learning_rate": args["learning_rate"],
        "batch_size": args["batch_size"],
        "num_updates": args["num_updates"]
        if args["num_updates"] is not None
        else 10 * args["num_chains"],
        "mult_optim": args["mult_optim"],
        "training_type": args["training_type"],
        "max_lr": args["max_lr"],
        "scale_lr": args["scale_lr"],
    }
    args_save = {
        "filename": args["filename"],
        "n_save": args["n_save"],
        "n_save_model": args["n_save_model"],
        "n_save_chain": args["n_save_chain"],
        "n_save_metric": args["n_save_metric"],
        "spacing": args["spacing"],
        "overwrite": args["overwrite"],
    }
    args_init = {
        "num_chains": args["num_chains"],
        "num_hiddens": args["num_hiddens"],
        "hidden_dims": args["hidden_dims"],
        "model_type": args["model_type"],
        "energy_type": args["energy_type"],
    }
    return (
        args_dataset,
        args_save,
        args_train,
        args_grad,
        args_ptt,
        args_torch,
        args_init,
    )


# @torch.no_grad
def main():
    print('Bonjour, je hardcode tt comme un psycopathe, déso pas déso nico, cf params hmc, lenet cnn, ... bcp')

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    args = set_args_default(args, default_args=default_args)
    args = set_args_default(args, default_args=default_args_ptt)
    args = match_args_dtype(args)
    (
        args_dataset,
        args_save,
        args_train,
        args_grad,
        args_ptt,
        args_torch,
        args_init,
    ) = process_args_ptt(args)
    checkpoints = get_checkpoints(
        num_updates=args_train["num_updates"],
        n_save=args_save["n_save"],
        spacing=args_save["spacing"],
    )
    # if args["variational"]:
    #     # load J1 J2
    #     temp = args["vartemp"]
    #     J1 = (
    #         torch.from_numpy(np.load(args["j1"])).to(args["dtype"]).to(args["device"])
    #         if args["j1"] is not None
    #         else torch.zeros(args["num_visibles"]).to(args["dtype"]).to(args["device"])
    #     )
    #     J2 = (
    #         torch.from_numpy(np.load(args["j2"])).to(args["dtype"]).to(args["device"])
    #         if args["j2"] is not None
    #         else torch.zeros(args["num_visibles"], args["num_visibles"])
    #         .to(args["dtype"])
    #         .to(args["device"])
    #     )
    #     J1 = J1 / temp
    #     J2 = J2 / temp
    #     num_visibles = args["num_visibles"]
    #     train_dataset = VarRBMDataset(
    #         J1=J1,
    #         J2=J2,
    #         num_visibles=num_visibles,
    #         num_chains=args["num_chains"],
    #         device=args["device"],
    #         dtype=args["dtype"],
    #         dataset_name=args["dataset"],
    #         variable_type="ising",
    #     )
    #     test_dataset = None
    # else:
    train_dataset, test_dataset = load_dataset(
        dataset_name=args_dataset["dataset_name"],
        test_dataset_name=args_dataset["test_dataset_name"],
        subset_labels=args_dataset["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        remove_duplicates=args["remove_duplicates"],
        **args_torch,
    )
    flags = ["checkpoint", "ptt"]
    if not args["restore"]:
        init_training_ptt(
            args,
            train_dataset,
            flags,
        )

    args = load_args_from_filename(args)
    print(args)
    args = set_args_default(args, default_args)
    if args["update"] is None:
        args["update"] = get_saved_updates(args["filename"])[-1]
    (
        params,
        parallel_chains,
        target_update,
        elapsed_time,
        train_dataset,
        test_dataset,
    ) = _restore_training(
        filename=args["filename"],
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_updates=args["num_updates"],
        target_update=args["update"],
        seed=args["seed"],
        train_size=args["train_size"],
        test_size=args["test_size"],
        device=args["device"],
        dtype=args["dtype"],
        map_model=map_model,
    )
    train_dataset.data_noise_std = float(args["data_noise_std"])
    test_dataset.data_noise_std = float(args["data_noise_std"])

    sampler = PTT.from_filename(
        filename=args["filename"],
        device=args_torch["device"],
        dtype=args_torch["dtype"],
        map_model=map_model,
    )
    sampler_kernel, sampler_kernel_params = get_sampler_kernel_args_ptt(args, params)
    sampler.sampler_kernel = sampler_kernel
    sampler.sampler_kernel_params = sampler_kernel_params
    while True:
        try:
            optimizer = setup_optim(args["optim"], args, params)
            args["scale_lr"] = (
                False  # We only need to scale the learning rate the first time
            )
            print(
                f"LEARNING RATE: {torch.tensor([opt.param_groups[0]['lr'] for opt in optimizer]).cpu().numpy()}"
            )
            pre_grad_update = build_pre_grad_update(
                optimizer=optimizer,
                lambda_l1=args["L1"],
                lambda_l2=args["L2"],
                normalize_grad=args["normalize_grad"],
                max_grad_norm=args["max_norm_grad"],
            )
            model_checkpoints = get_checkpoints(
                num_updates=args["num_updates"],
                n_save=args["n_save_model"] or args["n_save"],
                spacing=args["spacing"],
            )

            chain_checkpoints = get_checkpoints(
                num_updates=args["num_updates"],
                n_save=args["n_save_chain"] or args["n_save"],
                spacing=args["spacing"],
            )

            metric_checkpoints = get_checkpoints(
                num_updates=args["num_updates"],
                n_save=args["n_save_metric"] or args["n_save"],
                spacing=args["spacing"],
            )
            # # Create a wrapper to run cleanup silently during training
            # def pre_grad_update_with_cleanup(*func_args, **func_kwargs):
            #     pre_grad_update_base(*func_args, **func_kwargs)
            #     clean_old_checkpoints(args["filename"], keep_last=10)
            # print("SAMPLER",sampler)
            train(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                params=params,
                sampler=sampler,
                optimizer=optimizer,
                batch_size=args["batch_size"],
                centered=not (args["no_center"]),
                curr_update=target_update,
                pre_grad_update=pre_grad_update,
                elapsed_time=elapsed_time,
                num_updates=args["num_updates"],
                filename=args["filename"],
                model_checkpoints=model_checkpoints,
                chain_checkpoints=chain_checkpoints,
                metric_checkpoints=metric_checkpoints,
                # variational=args["variational"],
            )
            break
        except AcceptanceRateException:
            sampler, params, args, target_update = reset_training(args)


if __name__ == "__main__":
    main()
