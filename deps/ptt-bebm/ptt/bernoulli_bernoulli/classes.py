from __future__ import annotations

from typing import override

import numpy as np
import torch
from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.partition_function.ais import compute_partition_function_ais
from torch import Tensor

from ptt.bernoulli_bernoulli.implement import (
    _partition_function,
    _ptt_sampling,
)
from ptt.custom_fn import clone_dict
from ptt.generic.classes import PTT


class BBPTT(PTT):
    _v: Tensor
    _mv: Tensor
    _h: Tensor
    _mh: Tensor
    _list_model: list[BBRBM]

    @override
    def __len__(self):
        return self._weight_matrix.shape[0]

    @override
    def get_chains(self, index):
        return {
            "visible": self._v[index],
            "hidden": self._h[index],
            "visible_mag": self._mv[index],
            "hidden_mag": self._mh[index],
            "weights": torch.ones(
                self._v[index].shape[0], device=self._v.device, dtype=self._v.dtype
            ),
        }

    @override
    def get_model(self, index) -> BBRBM:
        return BBRBM(
            weight_matrix=self._weight_matrix[index],
            vbias=self._vbias[index],
            hbias=self._hbias[index],
            device=self.device,
            dtype=self.dtype,
        )

    @override
    def init_random_chains(self, num_chains: int):
        self._v = torch.zeros(
            self._weight_matrix.shape[0],
            num_chains,
            self._weight_matrix.shape[1],
            device=self.device,
            dtype=self.dtype,
        )
        self._h = torch.zeros(
            self._weight_matrix.shape[0],
            num_chains,
            self._weight_matrix.shape[2],
            device=self.device,
            dtype=self.dtype,
        )
        self._mv = torch.zeros(
            self._weight_matrix.shape[0],
            num_chains,
            self._weight_matrix.shape[1],
            device=self.device,
            dtype=self.dtype,
        )
        self._mh = torch.zeros(
            self._weight_matrix.shape[0],
            num_chains,
            self._weight_matrix.shape[2],
            device=self.device,
            dtype=self.dtype,
        )

        for i, m in enumerate(self._list_model):
            chain = m.init_chains(num_samples=num_chains)
            self._v[i] = chain["visible"]
            self._h[i] = chain["hidden"]
            self._mv[i] = chain["visible_mag"]
            self._mh[i] = chain["hidden_mag"]
            # self._weights[i] = chain["weights"]

    # @override
    def init_annealing_chains(self, num_chains, num_steps, start_v=None):
        super().init_annealing_chains(
            num_chains=num_chains, num_steps=num_steps, start_v=start_v
        )
        self._v = torch.stack([c["visible"] for c in self._chains])
        self._h = torch.stack([c["hidden"] for c in self._chains])
        self._mv = torch.stack([c["visible_mag"] for c in self._chains])
        self._mh = torch.stack([c["hidden_mag"] for c in self._chains])

    @override
    def sample(
        self,
        num_steps=None,
        **kwargs,
        # increment=None,
        # show_pbar=False,
        # show_acc_rate=False,
        # perform_swap=True,
    ):
        if "increment" in kwargs.keys():
            increment = kwargs["increment"]
        else:
            increment = self._increment
        if "show_pbar" in kwargs.keys():
            show_pbar = kwargs["show_pbar"]
        else:
            show_pbar = False
        if "perform_swap" in kwargs.keys():
            perform_swap = kwargs["perform_swap"]
        else:
            perform_swap = True
        if "show_acc_rate" in kwargs.keys():
            show_acc_rate = kwargs["show_acc_rate"]
        else:
            show_acc_rate = False
        if num_steps is None:
            num_swaps = int(self._num_swaps * np.sqrt(len(self._list_model)))
        else:
            num_swaps = num_steps
        if increment is None:
            increment = self._increment
        if True:
            self._v, self._h, self._mv, self._mh, self.acc_rates, self.index = (
                _ptt_sampling(
                    v=self._v,
                    weight_matrix=self._weight_matrix,
                    vbias=self._vbias,
                    hbias=self._hbias,
                    it_mcmc=num_swaps,
                    increment=increment,
                    pre_sampler=self._pre_sampler,
                    index=self.index,
                    show_pbar=show_pbar,
                    perform_swap=perform_swap,
                )
            )
        else:
            raise NotImplementedError
            # import jax
            # from ptt.bernoulli_bernoulli.implement import ptt_sampling_jax

            # v_jax, h_jax, mv_jax, mh_jax, acc_rates_jax = ptt_sampling_jax(
            #     key=jax.random.key(np.random.randint(0, 1000000)),
            #     v=jax.dlpack.from_dlpack(self._v),
            #     h=jax.dlpack.from_dlpack(self._h),
            #     mv=jax.dlpack.from_dlpack(self._mv),
            #     mh=jax.dlpack.from_dlpack(self._mh),
            #     weight_matrix=jax.dlpack.from_dlpack(self._weight_matrix),
            #     vbias=jax.dlpack.from_dlpack(self._vbias),
            #     hbias=jax.dlpack.from_dlpack(self._hbias),
            #     num_swaps=num_swaps,
            #     num_gibbs=increment,
            #     # pre_sampler=self._pre_sampler,
            #     # index=self.index,
            #     # show_pbar=show_pbar,
            #     # perform_swap=perform_swap,
            # )
            # self._v = torch.from_dlpack(v_jax)
            # self._h = torch.from_dlpack(h_jax)
            # self._mv = torch.from_dlpack(mv_jax)
            # self._mh = torch.from_dlpack(mh_jax)
            # self.acc_rates = torch.from_dlpack(acc_rates_jax)

    @override
    def compute_partition_function(self):
        if self._log_z_init is None:
            self._log_z_init = compute_partition_function_ais(
                self.get_chains(0)["visible"].shape[0], 5000, self.get_model(0)
            )
        self.log_z = _partition_function(
            tensor_visible=self._v,
            tensor_weight_matrix=self._weight_matrix,
            tensor_vbias=self._vbias,
            tensor_hbias=self._hbias,
            log_z_init=self._log_z_init,
        )
        return self.log_z

    @override
    def set_last_model(self, model):
        self._weight_matrix[-1] = model.weight_matrix.to(
            device=self.device, dtype=self.dtype
        )
        self._vbias[-1] = model.vbias.to(device=self.device, dtype=self.dtype)
        self._hbias[-1] = model.hbias.to(device=self.device, dtype=self.dtype)

    @override
    @property
    def num_chains(self) -> int:
        return self._v.shape[1]

    @override
    def set_chains(self, chains):
        assert len(chains) == len(self)
        # Initialize empty tensors to store in case
        self.init_random_chains(chains[0]["visible"].shape[0])
        for i, chain in enumerate(chains):
            self._v[i] = chain["visible"]
            self._h[i] = chain["hidden"]
            self._mv[i] = chain["visible_mag"]
            self._mh[i] = chain["hidden_mag"]
            # self._weights[i] = chain["weights"]

    @override
    def cut_sampler(self, index):
        self._weight_matrix = self._weight_matrix[index:]
        self._vbias = self._vbias[index:]
        self._hbias = self._hbias[index:]
        if self._v is not None:
            self._v = self._v[index:]
            self._h = self._h[index:]
            self._mv = self._mv[index:]
            self._mh = self._mh[index:]
        # self._weights = self._weights[index:]

    @override
    def pop_model(self, i):
        self._list_model.pop(i)
        self._weight_matrix = torch.cat(
            [self._weight_matrix[:i], self._weight_matrix[i + 1 :]]
        )
        self._vbias = torch.cat([self._vbias[:i], self._vbias[i + 1 :]])
        self._hbias = torch.cat([self._hbias[:i], self._hbias[i + 1 :]])
        self._v = torch.cat([self._v[:i], self._v[i + 1 :]])
        self._mv = torch.cat([self._mv[:i], self._mv[i + 1 :]])
        self._h = torch.cat([self._h[:i], self._h[i + 1 :]])
        self._mh = torch.cat([self._mh[:i], self._mh[i + 1 :]])
        # self._weights = torch.cat([self._weights[:i], self._weights[i+1:]])

    @override
    def set_list_model(self, list_model):
        super().set_list_model(list_model)
        self._weight_matrix = torch.stack([m.weight_matrix for m in self._list_model])
        self._vbias = torch.stack([m.vbias for m in self._list_model])
        self._hbias = torch.stack([m.hbias for m in self._list_model])

    @override
    def clone(self, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        list_model = [
            BBRBM(
                weight_matrix=self._weight_matrix[i].clone(),
                vbias=self._vbias[i].clone(),
                hbias=self._hbias[i].clone(),
            )
            for i in range(len(self))
        ]
        sampler = BBPTT(
            list_model=list_model,
            num_chains=self.num_chains,
            increment=self._increment,
            num_swaps=self._num_swaps,
            log_z_init=self._log_z_init,
            target_acc_rate=self.target_acc_rate,
            max_n_model=self.max_n_model,
            target_n_model=self.target_n_model,
            full_sampler=self.full_sampler,
            reservoir_size=self.reservoir_size,
            n_sample_steps=self.n_sample_steps,
            device=device,
            dtype=dtype,
        )
        sampler.set_pre_sampler(
            self._pre_sampler.clone(device=device, dtype=dtype)
            if self._pre_sampler is not None
            else None
        )
        if self._v is not None:
            chains = [clone_dict(self.get_chains(i)) for i in range(len(self))]
            sampler.set_chains(chains=chains)
        return sampler

    @override
    def insert_model(self, i, model, chains):
        if i < len(self):
            self._list_model.insert(i, model)
            self._weight_matrix = torch.cat(
                [
                    self._weight_matrix[:i],
                    model.weight_matrix.unsqueeze(0),
                    self._weight_matrix[i:],
                ]
            )
            self._vbias = torch.cat(
                [self._vbias[:i], model.vbias.unsqueeze(0), self._vbias[i:]]
            )
            self._hbias = torch.cat(
                [self._hbias[:i], model.hbias.unsqueeze(0), self._hbias[i:]]
            )
            self._v = torch.cat(
                [self._v[:i], chains["visible"].unsqueeze(0), self._v[i:]]
            )
            self._mv = torch.cat(
                [self._mv[:i], chains["visible_mag"].unsqueeze(0), self._mv[i:]]
            )
            self._h = torch.cat([self._h[:i], chains["hidden"].unsqueeze(0), self._h[i:]])
            self._mh = torch.cat(
                [self._mh[:i], chains["hidden_mag"].unsqueeze(0), self._mh[i:]]
            )
