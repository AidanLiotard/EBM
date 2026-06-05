import numpy as np
import torch
from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.bernoulli_bernoulli.implement import _compute_energy_visibles
from rbms.partition_function.ais import compute_partition_function_ais
from torch import Tensor
from tqdm.autonotebook import tqdm

from ptt.pre_sampler.classes import PreSampler


def _compute_energy_parallel(
    v: Tensor, weight_matrix: Tensor, vbias: Tensor, hbias: Tensor
):
    field = torch.bmm(v, vbias.view(vbias.shape[0], vbias.shape[1], 1)).squeeze()
    exponent = hbias.unsqueeze(1) + torch.bmm(v, weight_matrix)
    log_term = torch.where(exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent)
    return -field - log_term.sum(-1)


def _parallel_sampling(
    gibbs_steps: int, v: Tensor, weight_matrix: Tensor, vbias: Tensor, hbias: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mh = torch.sigmoid(
        torch.bmm(v, weight_matrix) + hbias.view(hbias.shape[0], 1, hbias.shape[1])
    )
    h = torch.bernoulli(mh)
    mv = torch.zeros_like(v)
    for i in range(gibbs_steps):
        mv = torch.sigmoid(
            torch.bmm(h, weight_matrix.permute(0, 2, 1))
            + vbias.view(vbias.shape[0], 1, vbias.shape[1])
        )
        v = torch.bernoulli(mv)
        mh = torch.sigmoid(
            torch.bmm(v, weight_matrix) + hbias.view(hbias.shape[0], 1, hbias.shape[1])
        )
        h = torch.bernoulli(mh)
    return v, mv, h, mh


def _compute_delta_energy(energy_model_conf: Tensor) -> Tensor:
    return (
        -energy_model_conf[0][1]
        + energy_model_conf[0][0]
        + energy_model_conf[1][1]
        - energy_model_conf[1][0]
    )


# @torch.compile(fullgraph=True)
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
            index_save = index[i].clone()
            index[i] = torch.where(swap, index[i + 1], index_save)
            index[i + 1] = torch.where(swap, index_save, index[i + 1])
        swap = swap.view(-1, 1).repeat(1, v.shape[2])
        v_save = v[i].clone()
        v[i] = torch.where(swap, v[i + 1], v_save)  # [torch.randperm(v.shape[1])]
        v[i + 1] = torch.where(swap, v_save, v[i + 1])  # [torch.randperm(v.shape[1])]
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
    for _ in range(0, it_mcmc):
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
            params=BBRBM(tensor_weight_matrix[0], tensor_vbias[0], tensor_hbias[0]),
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


# import jax
# import jax.numpy as jnp
# from jax import Array
# from jax.nn import sigmoid


# @jax.jit
# def sample_hiddens(
#     key: Array, v: Array, weight_matrix: Array, hbias: Array, beta: float = 1.0
# ):
#     mh = sigmoid(beta * (hbias + (v @ weight_matrix)))
#     h = jax.random.bernoulli(key, mh).astype(jnp.float32)
#     return h, mh


# @jax.jit
# def sample_visibles(
#     key: Array, h: Array, weight_matrix: Array, vbias: Array, beta: float = 1.0
# ):
#     mh = sigmoid(beta * (vbias + (h @ weight_matrix.T)))
#     h = jax.random.bernoulli(key, mh).astype(jnp.float32)
#     return h, mh


# @jax.jit
# def gibbs_step(
#     i,
#     arg_in,
# ):
#     (
#         key,
#         v,
#         h,
#         mv,
#         mh,
#         weight_matrix,
#         vbias,
#         hbias,
#     ) = arg_in
#     key, subkey = jax.random.split(key)
#     v, mv = sample_visibles(
#         subkey,
#         h,
#         weight_matrix,
#         vbias,
#     )

#     key, subkey = jax.random.split(key)
#     h, mh = sample_hiddens(
#         subkey,
#         v,
#         weight_matrix,
#         hbias,
#     )
#     return (key, v, h, mv, mh, weight_matrix, vbias, hbias)


# @jax.jit
# def gibbs_sampling(key, num_steps, v, h, mv, mh, weight_matrix, vbias, hbias):
#     arg_in = (key, v, h, mv, mh, weight_matrix, vbias, hbias)
#     (key, v, h, mv, mh, weight_matrix, vbias, hbias) = jax.lax.fori_loop(
#         0, num_steps, gibbs_step, arg_in
#     )
#     return v, h, mv, mh


# @jax.jit
# def compute_energy_visibles(
#     v: Array, vbias: Array, hbias: Array, weight_matrix: Array
# ) -> Array:
#     field = v @ vbias
#     exponent = hbias + (v @ weight_matrix)
#     log_term = jax.nn.softplus(exponent)
#     return -field - log_term.sum(1)


# @jax.jit
# def parallel_sampling(
#     key: Array,
#     num_steps: int,
#     v: Array,
#     h: Array,
#     mv: Array,
#     mh: Array,
#     weight_matrix: Array,
#     vbias: Array,
#     hbias: Array,
# ) -> tuple[Array, Array, Array, Array]:
#     tmp = jax.random.split(key, v.shape[0] + 1)
#     key = tmp[0]
#     subkeys = tmp[1:]
#     return jax.vmap(gibbs_sampling, in_axes=(0, None, 0, 0, 0, 0, 0, 0, 0))(
#         subkeys, num_steps, v, h, mv, mh, weight_matrix, vbias, hbias
#     )


# @jax.jit
# def swap_array(swap, arr_a, i, j):
#     tmp = jnp.copy(arr_a[i])
#     arr_a = arr_a.at[i].set(jnp.where(swap, arr_a[j], tmp))
#     arr_a = arr_a.at[j].set(jnp.where(swap, tmp, arr_a[j]))
#     return arr_a


# def swap_iter(
#     i: int,
#     arg_in: tuple[
#         Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
#     ],
# ):
#     (
#         key,
#         v,
#         h,
#         mv,
#         mh,
#         weight_matrix,
#         vbias,
#         hbias,
#         energies_next,
#         energy_curr,
#         acc_rates,
#     ) = arg_in
#     energy_next_model_next = compute_energy_visibles(
#         v[i + 1], vbias[i + 1], hbias[i + 1], weight_matrix[i + 1]
#     )
#     energy_curr_model_next = compute_energy_visibles(
#         v[i], vbias[i + 1], hbias[i + 1], weight_matrix[i + 1]
#     )
#     delta_energy = (
#         -energies_next[i] + energy_curr + energy_next_model_next - energy_curr_model_next
#     )
#     key, subkey = jax.random.split(key)
#     swap = jnp.exp(delta_energy) > jax.random.uniform(subkey, (v.shape[1],))
#     swap_vis = swap[:, None].repeat(v.shape[2], 1)
#     swap_hid = swap[:, None].repeat(h.shape[2], 1)
#     energy_curr = jnp.where(swap, energy_curr_model_next, energy_next_model_next)
#     v = swap_array(swap_vis, v, i, i + 1)
#     mv = swap_array(swap_vis, mv, i, i + 1)
#     h = swap_array(swap_hid, h, i, i + 1)
#     mh = swap_array(swap_hid, mh, i, i + 1)
#     curr_acc_rate = swap.sum() / v.shape[1]
#     # jax.debug.print("{curr_acc_rate}", curr_acc_rate=curr_acc_rate)
#     # acc_rates.at[i].set(swap.sum() / v.shape[1])
#     acc_rates = jnp.put(acc_rates, i, curr_acc_rate, inplace=False)
#     # assert curr_acc_rate == acc_rates[i].item()
#     return (
#         key,
#         v,
#         h,
#         mv,
#         mh,
#         weight_matrix,
#         vbias,
#         hbias,
#         energies_next,
#         energy_curr,
#         acc_rates,
#     )


# def _swap_config_parallel(
#     key: Array,
#     v: Array,
#     h: Array,
#     mv: Array,
#     mh: Array,
#     weight_matrix: Array,
#     vbias: Array,
#     hbias: Array,
# ) -> tuple[Array, Array, Array, Array, Array]:
#     energies_next = jax.vmap(compute_energy_visibles)(
#         v[1:], vbias[:-1], hbias[:-1], weight_matrix[:-1]
#     )
#     energy_curr = compute_energy_visibles(v[0], vbias[0], hbias[0], weight_matrix[0])
#     acc_rates = jnp.zeros(v.shape[0] - 1)
#     arg_in = (
#         key,
#         v,
#         h,
#         mv,
#         mh,
#         weight_matrix,
#         vbias,
#         hbias,
#         energies_next,
#         energy_curr,
#         acc_rates,
#     )
#     (
#         key,
#         v,
#         h,
#         mv,
#         mh,
#         weight_matrix,
#         vbias,
#         hbias,
#         energies_next,
#         energy_curr,
#         acc_rates,
#     ) = jax.lax.fori_loop(0, v.shape[0] - 1, swap_iter, arg_in)
#     return v, h, mv, mh, acc_rates


# def ptt_iter(i, arg_in):
#     (key, v, h, mv, mh, weight_matrix, vbias, hbias, num_gibbs, acc_rates) = arg_in
#     key, subkey = jax.random.split(key)
#     v, h, mv, mh, acc_rates = _swap_config_parallel(
#         subkey, v, h, mv, mh, weight_matrix, vbias, hbias
#     )
#     key, subkey = jax.random.split(key)
#     v, h, mv, mh = parallel_sampling(
#         subkey, num_gibbs, v, h, mv, mh, weight_matrix, vbias, hbias
#     )
#     return (key, v, h, mv, mh, weight_matrix, vbias, hbias, num_gibbs, acc_rates)


# @jax.jit
# def ptt_sampling_jax(
#     key: Array,
#     v: Array,
#     h: Array,
#     mv: Array,
#     mh: Array,
#     weight_matrix: Array,
#     vbias: Array,
#     hbias: Array,
#     num_swaps,
#     num_gibbs,
# ):
#     key, subkey = jax.random.split(key)
#     acc_rates = jnp.zeros(v.shape[0] - 1)
#     arg_in = (subkey, v, h, mv, mh, weight_matrix, vbias, hbias, num_gibbs, acc_rates)
#     (_, v, h, mv, mh, weight_matrix, vbias, hbias, num_gibbs, acc_rates) = (
#         jax.lax.fori_loop(0, num_swaps, ptt_iter, arg_in)
#     )
#     return v, h, mv, mh, acc_rates
