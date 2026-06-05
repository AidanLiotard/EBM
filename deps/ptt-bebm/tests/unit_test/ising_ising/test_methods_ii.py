import pytest
import torch

from ptt.custom_fn import clone_dict
def test_insert_model(sample_sampler_iirbm, sample_params_class_iirbm, sample_chains_iirbm):
    new_model = sample_params_class_iirbm
    sampler = sample_sampler_iirbm
    chains = sample_chains_iirbm
    model_prev = sampler.get_model(-2).clone()
    model_next = sampler.get_model(-1).clone()
    chains_prev = clone_dict(sampler.get_chains(-2))
    chains_next = clone_dict(sampler.get_chains(-1))
    sampler.insert_model(len(sampler)-1, new_model, chains)

    assert torch.equal(chains_prev["visible"], sampler.get_chains(-3)["visible"])
    assert torch.equal(chains_next["visible"], sampler.get_chains(-1)["visible"])
    assert torch.equal(chains["visible"], sampler.get_chains(-2)["visible"])

    assert model_prev == sampler.get_model(-3).clone() 
    assert model_next == sampler.get_model(-1).clone()
    assert new_model == sampler.get_model(-2).clone()

    assert model_prev == sampler._list_model[0]
    assert model_next == sampler._list_model[2]
    assert new_model == sampler._list_model[1]