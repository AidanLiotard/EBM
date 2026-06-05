import numpy as np
import torch

from rbms.classes import EBM
from ptt.pre_sampler.classes import PreSampler
from ptt.pre_sampler.rcm import BBRCM, PBRCM, IIRCM
from ptt.pre_sampler.reservoir import Reservoir

map_pre_sampler: dict[str, type[PreSampler]] = {
    "BBRCM": BBRCM,
    "IIRCM": IIRCM,
    "PBRCM": PBRCM,
    "IIRCM": IIRCM,
    "Reservoir": Reservoir,
}


def get_pre_sampler(
    named_params: dict[str, np.ndarray],
    ebm: EBM,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PreSampler | None:
    pre_sampler_type = str(
        named_params.pop("pre_sampler_type").astype(np.dtypes.StringDType)
    )
    if pre_sampler_type!='none':
        pre_sampler_type='Reservoir'

    if pre_sampler_type == "none":
        return None
    return map_pre_sampler[pre_sampler_type].set_named_parameters(
        ebm=ebm, named_params=named_params, device=device, dtype=dtype
    )
