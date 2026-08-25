import torch
import ai3
from ai3.errors import UnsupportedCallableError
from run import CONV2D_ALGOS_TO_USE
from test import compare_tensors
import model_zoo
import sys


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    with torch.inference_mode():
        for algo in CONV2D_ALGOS_TO_USE:
            try:
                ai3_model = ai3.convert(
                    module, {'conv2d': algo})
            except UnsupportedCallableError as e:
                print(f'  {e} so skipping')
                return
            output = ai3_model(input_data)
            compare_tensors(
                output, target,
                f'{name} swap backend using {algo}, {model_zoo.BATCH} samples')


def mha_runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    with torch.inference_mode():
        try:
            ai3_model = ai3.convert(module, {'mha': 'default'})
        except UnsupportedCallableError as e:
            print(f'  {e} so skipping')
            return
        output = ai3_model(input_data)
        compare_tensors(
            output, target,
            f'{name} convert mha, {model_zoo.BATCH} samples')


if __name__ == '__main__':
    model_zoo.from_args(runner, sys.argv)
