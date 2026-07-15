import torch
import ai3
from bench import predict_show_time
from test import compare_tensors
import model_zoo
import sys


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    torch_out = predict_show_time(
        module, input_data, name + ' torch')
    assert (isinstance(torch_out, torch.Tensor))

    ai3.swap_mha(module)
    ai3_out = predict_show_time(
        module, input_data, f'{name} ai3')
    compare_tensors(ai3_out, torch_out,
                    f'{name} ai3, {model_zoo.BATCH} samples',
                    print_pass=False, atol=1e-1)


if __name__ == '__main__':
    model_zoo.from_args(runner, sys.argv, model_zoo.MHA)
