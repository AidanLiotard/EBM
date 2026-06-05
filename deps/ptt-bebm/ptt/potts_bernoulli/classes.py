from typing import override

import numpy as np
import torch
from rbms.custom_fn import one_hot
from rbms.partition_function.ais import compute_partition_function_ais
from rbms.potts_bernoulli.classes import PBRBM
from torch import Tensor

from ptt.generic.classes import PTT
from ptt.custom_fn import clone_dict
from ptt.potts_bernoulli.implement import _partition_function, _ptt_sampling


class PBPTT(PTT):
    _v: Tensor
    _mv: Tensor
    _h: Tensor
    _mh: Tensor
    _list_model: list[PBRBM]
    _num_states: int

    def __init__(
        self,
        list_model: list[PBRBM],
        num_chains: int,
        increment: int,
        num_swaps: int,
        log_z_init: float,
        target_acc_rate: float,
        max_n_model: int,
        target_n_model: int,
        reservoir_size: int,
        n_sample_steps: int,
        full_sampler: bool,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self._num_states = list_model[0].num_states
        super().__init__(
            list_model,  # type: ignore
            num_chains,
            increment,
            num_swaps,
            target_acc_rate,
            max_n_model,
            target_n_model,
            full_sampler,
            reservoir_size,
            n_sample_steps,
            log_z_init,
            device,
            dtype,
        )

    @override
    def __len__(self):
        return self._weight_matrix.shape[0]

    @override
    def get_chains(self, index):
        return {
            "visible": (
                self._v[index]
                .reshape(self._v[index].shape[0], -1, self.num_states)
                .argmax(-1)
                .to(self.dtype)
            ),
            "hidden": self._h[index],
            "visible_mag": self._mv[index],
            "hidden_mag": self._mh[index],
            "weights": torch.ones(
                self._v[index].shape[0], device=self._v.device, dtype=self._v.dtype
            ),
        }

    @override
    def get_model(self, index):
        return PBRBM(
            weight_matrix=self._weight_matrix[index].reshape(
                -1, self.num_states, self._weight_matrix.shape[-1]
            ),
            vbias=self._vbias[index],
            hbias=self._hbias[index],
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def num_states(self):
        return self._num_states

    @override
    def set_chains(self, chains):
        assert len(chains) == len(self)
        # Initialize empty tensors to store in case
        self.init_random_chains(chains[0]["visible"].shape[0])
        for i, chain in enumerate(chains):
            self._v[i] = (
                one_hot(chain["visible"].to(torch.int64), num_classes=self.num_states)
                .view(chain["visible"].shape[0], -1)
                .to(self.dtype)
            )
            self._h[i] = chain["hidden"]
            self._mv[i] = chain["visible_mag"].reshape(chain["visible_mag"].shape[0], -1)
            self._mh[i] = chain["hidden_mag"]
            # self._weights[i] = chain["weights"]

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
        # self._weights = torch.zeros(
        #     self._weight_matrix.shape[0], num_chains, device=self.device, dtype=self.dtype
        # )
        for i, m in enumerate(self._list_model):
            chain = m.init_chains(num_samples=num_chains)
            self._v[i] = (
                one_hot(chain["visible"].to(torch.int64), num_classes=self.num_states)
                .view(chain["visible"].shape[0], -1)
                .to(self.dtype)
            )
            self._h[i] = chain["hidden"]
            self._mv[i] = chain["visible_mag"].reshape(chain["visible_mag"].shape[0], -1)
            self._mh[i] = chain["hidden_mag"]
            # self._weights[i] = chain["weights"]

    @override
    def init_annealing_chains(self, num_chains, num_steps, start_v=None):
        super().init_annealing_chains(num_chains, num_steps, start_v)
        self._v = torch.stack(
            [
                (
                    one_hot(c["visible"].to(torch.int64), num_classes=self.num_states)
                    .view(c["visible"].shape[0], -1)
                    .to(self.dtype)
                )
                for c in self._chains
            ]
        )
        self._h = torch.stack([c["hidden"] for c in self._chains])
        self._mv = torch.stack([c["visible_mag"] for c in self._chains])
        self._mh = torch.stack([c["hidden_mag"] for c in self._chains])
        # self._weights = torch.stack([c["weights"] for c in _chains])

    @override
    def sample(self, num_steps=None, **kwargs):
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
        self._v, self._h, self._mv, self._mh, self.acc_rates, self.index = _ptt_sampling(
            v=self._v,
            weight_matrix=self._weight_matrix,
            vbias=self._vbias,
            hbias=self._hbias,
            it_mcmc=num_swaps,
            num_states=self.num_states,
            increment=increment,
            pre_sampler=self._pre_sampler,
            index=self.index,
            show_pbar=show_pbar,
            perform_swap=perform_swap,
        )

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
        self._weight_matrix[-1] = model.weight_matrix.reshape(
            -1, model.weight_matrix.shape[-1]
        ).to(device=self.device, dtype=self.dtype)
        self._vbias[-1] = model.vbias.to(device=self.device, dtype=self.dtype)
        self._hbias[-1] = model.hbias.to(device=self.device, dtype=self.dtype)

    @override
    @property
    def num_chains(self) -> int:
        return self._v.shape[1]

    @override
    def cut_sampler(self, index):
        self._weight_matrix = self._weight_matrix[index:]
        self._vbias = self._vbias[index:]
        self._hbias = self._hbias[index:]
        self._list_model = self._list_model[index:]
        if self._v is not None:
            self._v = self._v[index:]
            self._h = self._h[index:]
            self._mv = self._mv[index:]
        # self._weights = self._weights[index:]

    @override
    def clone(self, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        list_model = [
            self.get_model(i).clone(device=device, dtype=dtype) for i in range(len(self))
        ]
        sampler = PBPTT(
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
        if self._v is not None:
            chains = [clone_dict(self.get_chains(i)) for i in range(len(self))]
            sampler.set_chains(chains=chains)

        return sampler

    @override
    def set_list_model(self, list_model):
        super().set_list_model(list_model)
        self._weight_matrix = torch.stack(
            [
                m.weight_matrix.reshape(-1, m.weight_matrix.shape[-1])
                for m in self._list_model
            ]
        )
        self._vbias = torch.stack([m.vbias for m in self._list_model])
        self._hbias = torch.stack([m.hbias for m in self._list_model])

    @override
    def insert_model(self, i, model, chains):
        if i < len(self):
            self._list_model.insert(i + 1, model)
            self._weight_matrix = torch.cat(
                [
                    self._weight_matrix[:i],
                    model.weight_matrix.reshape(
                        -1, model.weight_matrix.shape[-1]
                    ).unsqueeze(0),
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
                [
                    self._v[:i],
                    (
                        one_hot(
                            chains["visible"].to(torch.int64), num_classes=self.num_states
                        )
                        .view(chains["visible"].shape[0], -1)
                        .to(self.dtype)
                        .unsqueeze(0)
                    ),
                    self._v[i:],
                ]
            )
            self._mv = torch.cat(
                [self._mv[:i], chains["visible_mag"].unsqueeze(0), self._mv[i:]]
            )
            self._h = torch.cat([self._h[:i], chains["hidden"].unsqueeze(0), self._h[i:]])
            self._mh = torch.cat(
                [self._mh[:i], chains["hidden_mag"].unsqueeze(0), self._mh[i:]]
            )
            # self._weights = torch.cat(
            #     [self._weights[:i], chains["weights"].unsqueeze(0), self._weights[i:]]
            # )
