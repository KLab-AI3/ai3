import torch
from torch import nn
import ai3
from test import compare_tensors
from run import CONV2D_ALGOS_TO_USE
from typing import Union, Tuple


class Conv2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias, stride,
            padding, dilation, groups):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

    def forward(self, input):
        return self.conv(input)


def test(*, num_samples=None, input_channels: int, in_height: int, in_width: int,
         output_channels: int, kernel_height: int, kernel_width: int,
         with_bias: bool = False,
         padding: Union[str,
                        Union[int, Tuple[int, int]]] = 1,
         dilation: Union[int, Tuple[int, int]] = 1,
         stride: Union[int, Tuple[int, int]] = 1,
         groups: int = 1,
         test_name: str) -> None:
    dtype = torch.float32
    if num_samples:
        input = torch.randn(num_samples, input_channels, in_height,
                            in_width, dtype=dtype)
    else:
        input = torch.randn(input_channels, in_height,
                            in_width, dtype=dtype)

    orig = Conv2D(input_channels, output_channels,
                  (kernel_height, kernel_width),
                  with_bias, stride, padding, dilation, groups)
    torch_output = orig(input)

    for algo in CONV2D_ALGOS_TO_USE:
        model: ai3.Model = ai3.convert(orig, {'conv2d': algo})
        out = model.predict(
            input, out_type=torch.Tensor)
        if algo == 'metal':
            atol = 1e-1
        else:
            atol = None
        compare_tensors(
            out, torch_output, test_name + f' {algo}', atol=atol)


def main():
    print('CONV2D')

    test(input_channels=1,
         in_height=5,
         in_width=5,
         output_channels=1,
         num_samples=1,
         padding=0,
         kernel_height=4,
         kernel_width=4,
         test_name='bias no bias no padding')

    test(input_channels=1,
         in_height=100,
         in_width=150,
         output_channels=1,
         kernel_height=15,
         kernel_width=12,
         with_bias=True,
         padding=5,
         test_name='with bias')

    test(input_channels=1,
         in_height=7,
         in_width=7,
         output_channels=1,
         kernel_height=3,
         kernel_width=3,
         padding=0,
         with_bias=False,
         dilation=2,
         test_name='1d dilation')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         dilation=(1, 2),
         test_name='2d dilation')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         padding='same',
         dilation=(2, 2),
         test_name='same odd kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=8,
         kernel_width=4,
         with_bias=True,
         padding='same',
         dilation=(2, 2),
         test_name='same even kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=8,
         kernel_width=4,
         with_bias=True,
         padding='valid',
         dilation=(1, 2),
         test_name='valid even kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         padding='valid',
         dilation=(1, 2),
         test_name='valid odd kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         padding=5,
         test_name='1d padding')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         padding=(2, 5),
         test_name='2d padding')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         stride=2,
         test_name='1d stride')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         stride=(2, 3),
         test_name='2d stride')

    test(input_channels=1,
         in_height=10,
         in_width=15,
         output_channels=1,
         kernel_height=10,
         kernel_width=15,
         with_bias=True,
         test_name='kern.shape = input.shape')

    test(input_channels=3,
         in_height=50,
         in_width=150,
         output_channels=4,
         kernel_height=10,
         kernel_width=10,
         test_name='multi channel')

    test(input_channels=4,
         in_height=50,
         in_width=150,
         output_channels=6,
         kernel_height=10,
         kernel_width=10,
         with_bias=True,
         test_name='multi channel with bias')

    test(input_channels=4,
         in_height=50,
         in_width=150,
         output_channels=6,
         kernel_height=10,
         kernel_width=10,
         with_bias=True,
         test_name='multi channel with bias')

    test(num_samples=5,
         input_channels=4,
         in_height=50,
         in_width=150,
         output_channels=6,
         kernel_height=5,
         kernel_width=5,
         with_bias=True,
         test_name='batched multi channel with bias')


if __name__ == '__main__':
    main()
