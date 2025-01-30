import ai3
from . import conv2d, linear, avgpool2d, adaptiveavgpool2d, maxpool2d, mha, flatten, relu
import torch
from torch import nn

custom = 'custom'
custom_dict = {key: custom for key in ai3.DEFAULT_ALGOS}


def test(mod: nn.Module, input_shape: list[int], name: str):
    if isinstance(mod, mha.MHA):
        ai3.swap_mha(mod, custom)
        m = mod
    else:
        m = ai3.convert(mod, custom_dict)
    try:
        if isinstance(m, mha.MHA):
            m(torch.randn(input_shape),
              torch.randn(input_shape),
              torch.randn(input_shape))
        else:
            m(torch.randn(input_shape))
    except RuntimeError as e:
        error_message = str(e).lower()
        if "trying to use custom" in error_message and "when no implementation exists" in error_message:
            print(f'  correct error for nonexistant custom {name}')
        else:
            raise


def main():
    print('CUSTOM')
    test(conv2d.Conv2D(3, 16, 3, True, 1, 1, 1, 1), [3, 224, 224], 'conv2d')
    test(linear.Linear(3, 16, True), [10, 3], 'linear')
    test(flatten.Flatten(2, 3), [3, 6, 7, 8], 'flatten')
    test(relu.ReLU(), [10, 100], 'relu')
    test(
        avgpool2d.AvgPool2D(5, 2, 1, True, True, None),
        [3, 10, 100],
        'avgpool2d')
    test(adaptiveavgpool2d.AdaptiveAvgPool2D(
        1), [3, 10, 100], 'adaptiveavgpool2d')
    test(maxpool2d.MaxPool2D(4, 2, 2, 1, False), [3, 100, 100], 'maxpool2d')
    test(mha.MHA(64, 4, 64, 64, True, True, True, True), [10, 64], 'mha')


if __name__ == '__main__':
    main()
