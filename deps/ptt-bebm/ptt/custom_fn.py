import torch
from torch import Tensor


# @torch.compile(fullgraph=True)
def swap_chains(
    chain_1: dict[str, Tensor], chain_2: dict[str, Tensor], idx: Tensor
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    new_chain_1 = dict()
    new_chain_2 = dict()

    new_chain_1["weights"] = torch.where(
        idx, chain_2["weights"].squeeze(), chain_1["weights"].squeeze()
    ).unsqueeze(-1)
    new_chain_2["weights"] = torch.where(
        idx, chain_1["weights"].squeeze(), chain_2["weights"].squeeze()
    ).unsqueeze(-1)

    idx_vis = idx.unsqueeze(1).repeat(1, chain_1["visible"].shape[1])

    # if len(chain_1["visible_mag"].shape) > len(chain_1["visible"].shape):
    #     idx_vis_mean = idx_vis.repeat(1, chain_1["visible_mag"].shape[2]).reshape(
    #         chain_1["visible_mag"].shape
    #     )
    # else:
    #     idx_vis_mean = idx_vis

    # idx_hid = idx.unsqueeze(1).repeat(1, chain_1["hidden"].shape[1])
    perm_1 = torch.randperm(new_chain_1["weights"].shape[0])
    perm_2 = torch.randperm(new_chain_1["weights"].shape[0])
    new_chain_1["visible"] = torch.where(
        idx_vis, chain_2["visible"], chain_1["visible"]
    )  # [perm_1]
    new_chain_2["visible"] = torch.where(
        idx_vis, chain_1["visible"], chain_2["visible"]
    )  # [perm_2]

    # new_chain_1["visible_mag"] = torch.where(
    #     idx_vis_mean, chain_2["visible_mag"], chain_1["visible_mag"]
    # )
    # new_chain_2["visible_mag"] = torch.where(
    #     idx_vis_mean, chain_1["visible_mag"], chain_2["visible_mag"]
    # )

    # new_chain_1["hidden"] = torch.where(
    #     idx_hid, chain_2["hidden"], chain_1["hidden"]
    # )  # [perm_1]
    # new_chain_2["hidden"] = torch.where(
    #     idx_hid, chain_1["hidden"], chain_2["hidden"]
    # )  # [perm_2]

    # new_chain_1["hidden_mag"] = torch.where(
    #     idx_hid, chain_2["hidden_mag"], chain_1["hidden_mag"]
    # )
    # new_chain_2["hidden_mag"] = torch.where(
    #     idx_hid, chain_1["hidden_mag"], chain_2["hidden_mag"]
    # )

    return new_chain_1, new_chain_2


def clone_dict(d: dict[str, Tensor]) -> dict[str, Tensor]:
    res = {}
    for k in d.keys():
        res[k] = d[k].clone()
    return res


# @torch.compile(fullgraph=True)
def swap_tensor(
    v1: Tensor, v2: Tensor, swap_mask: Tensor, swap_only_v1: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Swap configurations between v_1 and v_2 on the first axis according to the boolean mask swap_mask

    Args:
        v1 (Tensor): shape (n, d)
        v2 (Tensor): shape (n, d)
        swap_mask (Tensor): shape (n, )

    Returns:
        tuple[Tensor, Tensor] v1, v2
    """
    swap_mask = swap_mask.view(-1, 1).repeat(1, v1.shape[1])
    if not (swap_only_v1):
        v_save = v1.clone()
        v1 = torch.where(swap_mask, v2, v_save)
        v2 = torch.where(swap_mask, v_save, v2)
    else:
        v1 = torch.where(swap_mask, v2, v1)
    return v1, v2