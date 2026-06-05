import torch
from rbms.bernoulli_bernoulli.classes import BBRBM
from torch import Tensor

from ptt.bernoulli_bernoulli.implement import _ptt_sampling
from ptt.pre_sampler import PreSampler


def ptt_sampling(
    list_params: list[BBRBM],
    chains: list[dict[str, Tensor]],
    index: Tensor | None,
    it_mcmc: int,
    pre_sampler: PreSampler | None = None,
    increment: int = 10,
    show_pbar: bool = True,
    show_acc_rate: bool = True,
):
    weight_matrix = torch.stack([p.weight_matrix for p in list_params])
    vbias = torch.stack([p.vbias for p in list_params])
    hbias = torch.stack([p.hbias for p in list_params])
    if index is not None:
        index = torch.stack([idx for idx in index])
    v = torch.stack([c["visible"] for c in chains])
    v, h, mv, mh, acc_rates, index = _ptt_sampling(
        v=v,
        weight_matrix=weight_matrix,
        vbias=vbias,
        hbias=hbias,
        it_mcmc=it_mcmc,
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
        c["visible"] = v[i].clone()
        c["hidden"] = h[i].clone()
        c["visible_mag"] = mv[i].clone()
        c["hidden_mag"] = mh[i].clone()
    if show_acc_rate:
        print("acc_rate: ", acc_rates)
    return chains, acc_rates, ret_index
