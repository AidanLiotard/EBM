import h5py
import numpy as np
import torch
from analysis.correlations import compute_2b_corr
from rbms.dataset.utils import convert_data
from rbms.utils import get_flagged_updates, get_saved_updates

from ptt.io import load_full_sampler_from_filename


def main(
    filename,
    out_file,
    train_dataset,
    test_dataset,
    increment,
    n_chains_trwa,
    n_chains_gen,
    n_samples,
    device,
    dtype,
):
    ptt_updates = get_flagged_updates(filename, "ptt")
    saved_updates = get_saved_updates(filename)
    ptt_updates = np.unique(np.append(ptt_updates, [saved_updates[-1]]))
    sampler = load_full_sampler_from_filename(
        filename,
        2000,
        False,
        device,
        dtype,
        num_steps=1,
        increment=increment,
        ptt_updates=ptt_updates,
    )
    num_visibles = sampler.get_model(-1).num_visibles

    # Check TRWA

    tau_int, tau_exp, C = sampler.trwa(
        num_chains=n_chains_trwa,
        increment=increment,
        filename=filename,
        force_recompute=False,
    )
    print(f"tau_int: {tau_int}")
    print(f"tau_exp: {tau_exp}")

    # Generate samples
    sampler.init_annealing_chains(n_chains_gen, 100)
    generated_samples = sampler.sample_large(
        num_samples=n_samples,
        num_steps_between=int(2 * tau_int + 1),
        num_steps_warmup=int(2 * tau_exp + 1),
        increment=increment,
        out_device="cpu",
    )

    corr_2b_ptt = torch.zeros(len(sampler), num_visibles, num_visibles)
    for i in range(len(sampler)):
        corr_2b_ptt[i] = compute_2b_corr(
            convert_data["categorical"]["bernoulli"](
                # generated_samples["visible"][i].cuda()
                generated_samples["visible"][i].to(args["device"])
            ).cpu()
        )

    # Compute LL
    train_ll = sampler.get_ll(train_dataset.data, train_dataset.weights)
    test_ll = sampler.get_ll(test_dataset.data, test_dataset.weights)
    # Save results in new file
    with h5py.File(out_file, "w") as f:
        f["samples"] = generated_samples["visible"].cpu().numpy()
        f["corr_2b"] = corr_2b_ptt.cpu().numpy()
        f["train_ll"] = train_ll.cpu().numpy()
        f["test_ll"] = test_ll.cpu().numpy()
        f["ptt_updates"] = ptt_updates
