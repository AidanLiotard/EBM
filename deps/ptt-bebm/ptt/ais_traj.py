from __future__ import annotations

import numpy as np
import torch
from rbms.classes import EBM, Sampler
from rbms.partition_function.ais import compute_partition_function_ais, update_weights_ais
from rbms.utils import compute_log_likelihood
from torch import Tensor

from ptt.utils import compute_ess, systematic_resampling


class AISTraj(Sampler):
    def __init__(
        self,
        params: EBM,
        chains: dict[str, Tensor],
        num_steps: int,
        beta: float = 1,
        **kwargs,
    ):
        self.name = "PCD"
        self.chains = chains
        self.params = params
        self.beta = beta
        self.num_steps = num_steps
        self.flags = []
        self.chains_ll_ais = params.sample_state(self.chains, 100, beta)
        self.log_weights = torch.zeros(
            chains["visible"].shape[0],
            device=chains["visible"].device,
            dtype=chains["visible"].dtype,
        )
        self.log_z_init_ais = compute_partition_function_ais(1000, 5000, self.params)
        self.log_z_ais = self.log_z_init_ais

    def get_conf_grad(self, batch: Tensor):
        self.sample(num_steps=None)
        return self.chains

    def sample(self, num_steps: int | None, **kwargs):
        self.chains = self.params.sample_state(
            chains=self.chains, n_steps=self.num_steps, beta=self.beta
        )

    @torch.compiler.disable
    def named_parameters(self):
        params_dict = self.params.named_parameters()
        params_dict["model_type"] = np.asarray(self.params.name, dtype="T")
        params_dict["sampler_type"] = np.asarray(self.name, dtype="T")
        params_dict["parallel_chains"] = self.chains["visible"].cpu().numpy()
        params_dict["beta"] = np.asarray(self.beta)
        params_dict["num_steps"] = np.asarray(self.num_steps)
        params_dict["log_weights"] = self.log_weights.cpu().numpy()
        params_dict["chains_ll_ais"] = self.chains_ll_ais["visible"].cpu().numpy()
        params_dict["log_z_ais"] = np.asarray(self.log_z_ais)
        params_dict["log_z_init_ais"] = np.asarray(self.log_z_init_ais)
        return params_dict

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        map_model: dict[str, type[EBM]],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> AISTraj:
        names = [
            "model_type",
            "chains",
            "beta",
            "num_steps",
            "log_weights",
            "chains_ll_ais",
            "log_z_ais",
            "log_z_init_ais",
        ]
        for k in names:
            if k not in named_params.keys():
                raise ValueError(
                    f"""Dictionary params missing key '{k}'\n Provided keys : {named_params.keys()}\n Expected keys: {names}"""
                )
        model_type = str(named_params.pop("model_type"))
        chains_visible = torch.from_numpy(named_params.pop("parallel_chains")).to(
            device=device, dtype=dtype
        )
        chains_ll_ais_visible = torch.from_numpy(named_params.pop("chains_ll_ais")).to(
            device=device, dtype=dtype
        )
        beta = float(named_params.pop("beta"))
        num_steps = int(named_params.pop("num_steps"))
        # There should only remain the keys for the model loading
        params = map_model[model_type].set_named_parameters(
            named_params=named_params, device=device, dtype=dtype
        )
        chains = params.init_chains(chains_visible.shape[0], start_v=chains_visible)
        chains_ll_ais = params.init_chains(
            chains_ll_ais_visible.shape[0], start_v=chains_ll_ais_visible
        )
        log_z_ais = named_params.pop("log_z_ais").item()
        log_z_init_ais = named_params.pop("log_z_init_ais").item()
        sampler = AISTraj(params=params, chains=chains, num_steps=num_steps, beta=beta)
        sampler.log_weights = torch.from_numpy(named_params.pop("log_weights")).to(
            device=device, dtype=dtype
        )
        sampler.chains_ll_ais = chains_ll_ais
        sampler.log_z_ais = log_z_ais
        sampler.log_z_init_ais = log_z_init_ais
        return sampler

    def post_grad_update(self, params: EBM):
        self.log_weights, self.chains_ll_ais = update_weights_ais(
            self.params,
            params,
            self.chains_ll_ais,
            log_weights=self.log_weights,
            n_steps=1,
        )
        self.log_z_ais = (
            torch.logsumexp(self.log_weights, 0)
            - np.log(self.chains_ll_ais["visible"].shape[0])
            + self.log_z_init_ais
        ).item()
        ess = compute_ess(self.log_weights)
        if ess < 0.3:
            # chains_ll_ais = clone_dict(parallel_chains)
            self.chains_ll_ais = systematic_resampling(
                self.chains_ll_ais, self.log_weights
            )
            self.log_z_init_ais = self.log_z_ais
            self.log_weights = torch.zeros_like(self.log_weights)
        self.params = params.clone()

    def get_metrics_display(self, metrics, **kwargs):
        train_batch = kwargs["train_dataset"].batch(self.chains["visible"].shape[0])
        test_batch = kwargs["test_dataset"].batch(self.chains["visible"].shape[0])

        metrics["LL train"] = compute_log_likelihood(
            v_data=train_batch["data"],
            w_data=train_batch["weights"],
            params=self.params,
            log_z=self.log_z_ais,
        )
        metrics["LL test"] = compute_log_likelihood(
            v_data=test_batch["data"],
            w_data=test_batch["weights"],
            params=self.params,
            log_z=self.log_z_ais,
        )
        metrics["ess"] = compute_ess(self.log_weights)
        return metrics

    def get_metrics_save(self):
        return {"log_z": self.log_z_ais}

    def pre_grad_update(self):
        pass
