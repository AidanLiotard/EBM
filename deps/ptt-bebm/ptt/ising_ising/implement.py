import numpy as np
import torch
from rbms.custom_fn import log2cosh
from rbms.ising_ising.classes import IIRBM
from rbms.ising_ising.implement import _compute_energy_visibles
from rbms.partition_function.ais import compute_partition_function_ais
from torch import Tensor
from tqdm.autonotebook import tqdm

from ptt.pre_sampler.classes import PreSampler


@torch.jit.script
def _compute_energy_parallel(
    v: Tensor, weight_matrix: Tensor, vbias: Tensor, hbias: Tensor
):
    field = torch.bmm(v, vbias.view(vbias.shape[0], vbias.shape[1], 1)).squeeze()
    exponent = hbias.unsqueeze(1) + torch.bmm(v, weight_matrix)
    log_term = log2cosh(exponent)
    return -field - log_term.sum(-1)


# @torch.jit.script
def _parallel_sampling(
    gibbs_steps: int, v: Tensor, weight_matrix: Tensor, vbias: Tensor, hbias: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    effective_field_h = torch.bmm(v, weight_matrix) + hbias.view(
        hbias.shape[0], 1, hbias.shape[1]
    )
    mh = torch.tanh(effective_field_h)
    h = 2 * torch.bernoulli(torch.sigmoid(2 * effective_field_h)) - 1
    mv = torch.zeros_like(v)
    for i in range(gibbs_steps):
        effective_field_v = torch.bmm(h, weight_matrix.permute(0, 2, 1)) + vbias.view(
            vbias.shape[0], 1, vbias.shape[1]
        )
        mv = torch.tanh(effective_field_v)
        v = 2 * torch.bernoulli(torch.sigmoid(2 * effective_field_v)) - 1
        effective_field_h = torch.bmm(v, weight_matrix) + hbias.view(
            hbias.shape[0], 1, hbias.shape[1]
        )
        mh = torch.tanh(effective_field_h)
        h = 2 * torch.bernoulli(torch.sigmoid(2 * effective_field_h)) - 1
    return v, mv, h, mh


@torch.jit.script
def _compute_delta_energy(energy_model_conf: Tensor) -> Tensor:
    return (
        -energy_model_conf[0][1]
        + energy_model_conf[0][0]
        + energy_model_conf[1][1]
        - energy_model_conf[1][0]
    )


# @torch.compile(fullgraph=True)
# @torch.jit.script
def swap_config_parallel(
    v: Tensor,
    weight_matrix: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    index: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    energies_next = _compute_energy_parallel(
        v[1:], weight_matrix[:-1], vbias[:-1], hbias[:-1]
    )

    n_models = v.shape[0]
    acc_rate = torch.zeros(n_models - 1, device=v.device)
    n_chains = v.shape[1]
    energies = torch.zeros(2, 2, n_chains, device=v.device)
    energies[0][0] = _compute_energy_visibles(v[0], vbias[0], hbias[0], weight_matrix[0])
    for i in range(n_models - 1):
        energies[0][1] = energies_next[i]
        # energies[0][1] = _compute_energy_visibles(
        #     v[i + 1], vbias[i], hbias[i], weight_matrix[i]
        # )
        # energies[0][0] = _compute_energy_visibles(
        #     v[i], vbias[i], hbias[i], weight_matrix[i]
        # )

        energies[1][0] = _compute_energy_visibles(
            v[i], vbias[i + 1], hbias[i + 1], weight_matrix[i + 1]
        )
        energies[1][1] = _compute_energy_visibles(
            v[i + 1], vbias[i + 1], hbias[i + 1], weight_matrix[i + 1]
        )
        delta_energy = _compute_delta_energy(energies)
        swap = torch.exp(delta_energy) > torch.rand(
            size=(n_chains,), device=delta_energy.device
        )
        energies[0][0] = torch.where(swap, energies[1][0], energies[1][1])
        acc_rate[i] = swap.sum() / n_chains

        if index is not None:
            swapped_index_0 = torch.where(swap, index[i + 1], index[i])
            swapped_index_1 = torch.where(swap, index[i], index[i + 1])
            index[i] = swapped_index_0
            index[i + 1] = swapped_index_1
            # index_save = index[i].clone()
            # index[i] = torch.where(swap, index[i + 1], index_save)
            # index[i + 1] = torch.where(swap, index_save, index[i + 1])
        swap = swap.view(-1, 1).repeat(1, v.shape[2])
        perm_1 = torch.randperm(v.shape[1])
        perm_2 = torch.randperm(v.shape[1])
        v_save = v[i].clone()
        v[i] = torch.where(swap, v[i + 1], v_save)[perm_1]

        v[i + 1] = torch.where(swap, v_save, v[i + 1])[perm_2]
        if index is not None:
            index[i] = index[i][perm_1]
            index[i + 1] = index[i + 1][perm_2]
    return v, acc_rate, index


# @torch.compile(fullgraph=True)
# @torch.no_grad
def _ptt_sampling(
    v: Tensor,
    weight_matrix: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    it_mcmc: int,
    increment: int = 10,
    pre_sampler: PreSampler | None = None,
    show_pbar: bool = True,
    perform_swap: bool = True,
    index: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    n_chains = v.shape[1]
    
    if show_pbar:
        pbar = tqdm(total=it_mcmc, leave=False)
    for it in range(0, it_mcmc):
        if show_pbar:
            pbar.update(1)
        if not perform_swap:
            save_v = v.clone()
        v, acc_rate, index = swap_config_parallel(v, weight_matrix, vbias, hbias, index)

        if not perform_swap:
            v = save_v
        v, mv, h, mh = _parallel_sampling(increment, v, weight_matrix, vbias, hbias)

        if pre_sampler is not None:
            pre_sampler.sample(num_samples=n_chains)
            swap_mask = pre_sampler.compute_swap_acc(visible_conf=v[0])
            v[0] = pre_sampler.perform_swap(v[0], swap_mask=swap_mask)
    if show_pbar:
        pbar.close()
    return v, h, mv, mh, acc_rate, index


# @torch.jit.script
def _partition_function(
    tensor_visible: Tensor,
    tensor_weight_matrix: Tensor,
    tensor_vbias: Tensor,
    tensor_hbias: Tensor,
    log_z_init: float | None = None,
) -> Tensor:
    if log_z_init is None:
        # Estimate the first log Z using AIS
        # Should work if this distribution is not multimodal
        log_z_init = compute_partition_function_ais(
            num_chains=5000,
            num_beta=1000,
            params=IIRBM(tensor_weight_matrix[0], tensor_vbias[0], tensor_hbias[0]),
        )

    logz = log_z_init
    logZ = torch.zeros(tensor_weight_matrix.shape[0])
    logZ[0] = log_z_init

    E0 = _compute_energy_parallel(
        v=tensor_visible,
        weight_matrix=tensor_weight_matrix,
        vbias=tensor_vbias,
        hbias=tensor_hbias,
    )
    E1 = _compute_energy_parallel(
        v=tensor_visible[:-1],
        weight_matrix=tensor_weight_matrix[1:],
        vbias=tensor_vbias[1:],
        hbias=tensor_hbias[1:],
    )

    for idx in range(tensor_weight_matrix.shape[0] - 1):
        c0 = torch.logsumexp(-E1[idx] + E0[idx], dim=0) - np.log(
            tensor_visible.shape[1]  # num_samples
        )
        logz += c0
        logZ[idx + 1] = logz
    return logZ
