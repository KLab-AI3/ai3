import torch
from torch import nn
from test.unit import pooling_poss_output_size
from ai3 import utils
from test import compare_tensors
import ai3
from typing import Union, Sequence, Optional


class MaxPool2D(nn.Module):
    def __init__(self, kernel_shape, stride, padding, dilation, ceil_mode):
        super(MaxPool2D, self).__init__()
        self.max2d = nn.MaxPool2d(
            kernel_shape, dilation=dilation, padding=padding, stride=stride,
            ceil_mode=ceil_mode)

    def forward(self, input):
        return self.max2d(input)


def test(*, input_channels: int, in_height: int, in_width: int,
         kernel_height: int, kernel_width: int,
         padding: Union[int, Sequence[int]] = 0,
         dilation: Union[int, Sequence[int]] = 1,
         stride: Optional[Union[int,
                                Sequence[int]]] = None,
         ceil_mode: bool = False,
         ceil_mode_note_height: bool = False,
         ceil_mode_note_width: bool = False,
         test_name: str) -> None:
    input = torch.randn(input_channels, in_height,
                        in_width, dtype=torch.float32)
    kernel_shape = (kernel_height, kernel_width)

    if stride is None:
        stride = kernel_shape

    if ceil_mode_note_height or ceil_mode_note_width:
        pos = pooling_poss_output_size(
            in_height, in_width, padding, stride, kernel_height, kernel_width,
            ceil_mode)
        stride = utils.make_2d(stride)
        padding = utils.make_2d(padding)
        assert (((pos[0] - 1) * stride[0] >= in_height +
                padding[0]) == ceil_mode_note_height)
        assert (((pos[1] - 1) * stride[1] >= in_width +
                padding[1]) == ceil_mode_note_width)

    orig = MaxPool2D(kernel_shape, stride, padding, dilation, ceil_mode)
    torch_output = orig(input)

    model = ai3.convert(orig)
    ai3_output = model.predict(input)
    compare_tensors(
        ai3_output, torch_output, test_name)


def main():
    print('MAX POOL 2D')
    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         dilation=(2, 2),
         test_name='same odd kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=8,
         kernel_width=4,
         dilation=(2, 2),
         test_name='same even kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=8,
         kernel_width=4,
         dilation=(1, 2),
         test_name='valid even kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         dilation=(1, 2),
         test_name='valid odd kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         dilation=(1, 2),
         test_name='2d dilation')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         dilation=3,
         test_name='1d dilation')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         padding=2,
         test_name='1d padding')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=10,
         padding=(2, 5),
         test_name='2d padding')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         stride=2,
         test_name='1d stride')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         stride=(2, 3),
         test_name='2d stride')

    test(input_channels=1,
         in_height=5,
         in_width=5,
         kernel_height=3,
         kernel_width=3,
         test_name='basic')

    test(input_channels=1,
         in_height=10,
         in_width=15,
         kernel_height=10,
         kernel_width=15,
         test_name='kern.shape = input.shape')

    test(input_channels=3,
         in_height=50,
         in_width=150,
         kernel_height=10,
         kernel_width=10,
         test_name='multi channel')

    test(input_channels=3,
         in_height=85,
         in_width=85,
         kernel_height=5,
         kernel_width=5,
         ceil_mode=True,
         test_name='ceil mode but ceil has no effect on output size')

    test(input_channels=3,
         in_height=85,
         in_width=85,
         kernel_height=7,
         kernel_width=7,
         padding=(3, 3),
         ceil_mode=True,
         test_name='ceil mode with padding')

    test(
        input_channels=3, in_height=6, in_width=6, stride=(4, 4),
        padding=(1, 1),
        kernel_height=2, kernel_width=2, ceil_mode=True,
        ceil_mode_note_height=True, ceil_mode_note_width=True,
        test_name='ceil mode with note, https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html')

    test(
        input_channels=3, in_height=6, in_width=40, stride=(4, 5),
        padding=(1, 1),
        kernel_height=2, kernel_width=5, ceil_mode=True,
        ceil_mode_note_height=True, ceil_mode_note_width=False,
        test_name='ceil mode with note for height, https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html')

    test(
        input_channels=3, in_height=40, in_width=6, stride=(5, 4),
        padding=(1, 1),
        kernel_height=5, kernel_width=2, ceil_mode=True,
        ceil_mode_note_height=False, ceil_mode_note_width=True,
        test_name='ceil mode with note for width, https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html')


if __name__ == '__main__':
    main()
