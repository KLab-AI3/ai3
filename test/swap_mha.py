import torch
import ai3
import model_zoo
import sys
from test import compare_tensors


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    with torch.inference_mode():
        ai3.swap_mha(module)
        output = module(input_data)
        compare_tensors(
            output, target,
            f'{name} swap mha, {model_zoo.BATCH} samples')


if __name__ == '__main__':
    model_zoo.from_args(runner, sys.argv, model_zoo.MHA)
