import torch
import ai3
from ai3.errors import UnsupportedCallableError
import models
import sys
from copy import copy, deepcopy
from test import compare_tensors


def alter_padding(module, target_padding=(100, 100)):
    for c in module.children():
        if isinstance(c, (torch.nn.Conv2d, ai3.swap_torch.Conv2D)):
            assert c.padding != target_padding
            c.padding = target_padding
        elif len(list(c.children())) > 0:
            alter_padding(c, target_padding)


def alter_orig_after_copy_ensure_same(
        orig, input_data: torch.Tensor, mes: str):
    target = orig(input_data)
    cpy = copy(orig)
    alter_padding(orig)
    cpy_out = cpy(input_data)
    compare_tensors(cpy_out, target, mes)
    return cpy_out


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    torch_cpy_out = alter_orig_after_copy_ensure_same(
        deepcopy(module), input_data, f"{name} torch after copying then altering orig")
    try:
        ai3_model = ai3.swap_backend(module)
    except UnsupportedCallableError as e:
        print(f"  {e} so skipping")
        return
    sb_out = ai3_model(input_data)
    compare_tensors(
        sb_out, target,
        f"{name} swap backend, not copied, {models.BATCH} samples")
    ai3.swap_conv2d(module)
    sc_out = module(input_data)
    compare_tensors(
        sc_out, target,
        f"{name} swap conv2d, not copied, {models.BATCH} samples")

    ai3_sc_cpy_out = alter_orig_after_copy_ensure_same(
        module, input_data, f"{name} ai3 after copying then altering orig")
    compare_tensors(
        ai3_sc_cpy_out, torch_cpy_out,
        f"{name} torch copied and ai3 sc copied, {models.BATCH} samples")

    ai3_sb_cpy = copy(ai3_model)
    ai3_sb_cpy_out = ai3_sb_cpy(input_data)
    compare_tensors(
        ai3_sb_cpy_out, torch_cpy_out,
        f"{name} torch copied and ai3 sb copied, {models.BATCH} samples")


if __name__ == "__main__":
    models.from_args(runner, sys.argv)
