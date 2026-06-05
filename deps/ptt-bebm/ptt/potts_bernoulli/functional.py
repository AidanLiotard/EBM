import torch
from rbms.custom_fn import one_hot
from rbms.potts_bernoulli.classes import PBRBM
from torch import Tensor

from ptt.potts_bernoulli.implement import _ptt_sampling
from ptt.pre_sampler import PreSampler


def ptt_sampling(
    list_params: list[PBRBM],
    chains: list[dict[str, Tensor]],
    index: Tensor | None,
    it_mcmc: int,
    pre_sampler: PreSampler | None = None,
    increment: int = 10,
    show_pbar: bool = True,
    show_acc_rate: bool = True,
):
    dtype = list_params[0].weight_matrix.dtype
    num_states = list_params[0].weight_matrix.shape[1]
    weight_matrix = torch.stack(
        [p.weight_matrix.reshape(-1, p.weight_matrix.shape[-1]) for p in list_params]
    )
    vbias = torch.stack([p.vbias for p in list_params])
    hbias = torch.stack([p.hbias for p in list_params])
    if index is not None:
        index = torch.stack([idx for idx in index])
    v = torch.stack(
        [
            one_hot(c["visible"].to(torch.int64), num_classes=num_states)
            .view(c["visible"].shape[0], -1)
            .to(dtype)
            for c in chains
        ]
    )
    v, h, mv, mh, acc_rates, index = _ptt_sampling(
        v=v,
        weight_matrix=weight_matrix,
        vbias=vbias,
        hbias=hbias,
        it_mcmc=it_mcmc,
        num_states=list_params[0].num_states,
        increment=increment,
        pre_sampler=pre_sampler,
        show_pbar=show_pbar,
        index=index,
    )
    ret_index = None
    if index is not None:
        ret_index = []
        for i in range(len(index)):
            ret_index.append(index[i])
    for i, c in enumerate(chains):
        c["visible"] = v[i].reshape(*c["visible"].shape, -1).argmax(-1)
        c["hidden"] = h[i].clone()
        c["visible_mag"] = mv[i].clone()
        c["hidden_mag"] = mh[i].clone()
    if show_acc_rate:
        print("acc_rate: ", acc_rates)
    return chains, acc_rates, ret_index
