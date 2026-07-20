import torch
from .models import *
from typing import Sequence

GROUPED_CONVOLUTION = False
BATCH = 2

CONV2D = 'conv2d'
MHA = 'mha'

MHA_MODELS = ['manual_mha', 'visiontransformer']


def from_args(runner, args, op=CONV2D):
    if len(args) > 1:
        for arg in args[1:]:
            run_on(runner, arg, op)
    else:
        run_on(runner, op=op)


def run_on(runner, name=None, op=CONV2D):
    if not name:
        run_all(runner, op)
    else:
        model_func = globals().get(name)
        if model_func:
            model, input_shape, name = model_func()
            wrapped_run(
                model, input_shape, name, runner, op)
        else:
            print(f'Invalid model {name}')


def check_mod(module: torch.nn.Module):
    found_conv2d = False
    grouped_conv = False
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Conv2d):
            found_conv2d = True
            if submodule.groups > 1:
                grouped_conv = True

    return grouped_conv, found_conv2d


def has_mha(module: torch.nn.Module):
    return any(isinstance(submodule, torch.nn.MultiheadAttention)
               for submodule in module.modules())


def wrapped_run(
        module: torch.nn.Module, input_sample_shape: Sequence[int],
        name, runner, op=CONV2D):
    name = name.upper()
    print(f'{name}')
    module.eval()
    if op == MHA:
        if not has_mha(module):
            print(f'{name} does not use multihead attention')
            return
        runner(module, torch.randn(BATCH, *input_sample_shape), name)
        return
    (needs_groups, has_conv) = check_mod(module)
    if needs_groups and not GROUPED_CONVOLUTION:
        print(
            f'  skipping {name} as it requires groups > 1')
    elif has_conv:
        runner(module, torch.randn(
            BATCH, *input_sample_shape), name)
    else:
        print(f'{name} does not use convolution')


def run_all(runner, op=CONV2D):
    if op == MHA:
        for name in MHA_MODELS:
            wrapped_run(*globals()[name](), runner, op)
        return
    wrapped_run(*alexnet(), runner)
    wrapped_run(*convnext(), runner)
    wrapped_run(*densenet(), runner)
    wrapped_run(*efficientnet(), runner)
    wrapped_run(*googlenet(), runner)
    wrapped_run(*inception(), runner)
    wrapped_run(*maxvit(), runner)
    wrapped_run(*mnasnet(), runner)
    wrapped_run(*mobilenet(), runner)
    wrapped_run(*regnet(), runner)
    wrapped_run(*resnet(), runner)
    wrapped_run(*shufflenet(), runner)
    wrapped_run(*manual_conv2d(), runner)
    wrapped_run(*squeezenet(), runner)
    wrapped_run(*swintransformer(), runner)
    wrapped_run(*vgg16(), runner)
    wrapped_run(*visiontransformer(), runner)
