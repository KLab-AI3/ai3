import torch
from test.unit import pooling_poss_output_size
from torch import nn
import ai3
from ai3 import utils
from test import compare_tensors
from typing import Union, Sequence, Optional


class AvgPool2D(nn.Module):
    def __init__(self, kernel_shape, stride, padding,
                 ceil_mode,
                 count_include_pad,
                 divisor_override):
        super(AvgPool2D, self).__init__()
        self.avgpool2d = nn.AvgPool2d(kernel_shape, stride, padding, ceil_mode,
                                      count_include_pad, divisor_override)

    def forward(self, input):
        return self.avgpool2d(input)


def test(*, input_channels: int, in_height: int, in_width: int,
         kernel_height: int, kernel_width: int,
         padding: Union[int, Sequence[int]] = 0,
         stride: Optional[Union[int,
                                Sequence[int]]] = None,
         ceil_mode: bool = False,
         ceil_mode_note_height: bool = False,
         ceil_mode_note_width: bool = False,
         count_include_pad=True,
         divisor_override: Optional[int] = None,
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

    orig = AvgPool2D(kernel_shape, stride, padding, ceil_mode,
                     count_include_pad, divisor_override)
    torch_output = orig(input)
    model = ai3.convert(orig)
    ai3_output = model.predict(input)
    compare_tensors(
        ai3_output, torch_output, test_name)


def main():
    print('AVG POOL 2D')
    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         test_name='basic')

    test(input_channels=3,
         in_height=35,
         in_width=58,
         kernel_height=7,
         kernel_width=5,
         ceil_mode=True,
         count_include_pad=False,
         test_name='ceil mode')

    test(input_channels=3,
         in_height=85,
         in_width=85,
         kernel_height=5,
         kernel_width=5,
         ceil_mode=True,
         count_include_pad=False,
         test_name='ceil mode but ceil has no effect on output size')

    test(input_channels=3,
         in_height=85,
         in_width=85,
         kernel_height=7,
         kernel_width=7,
         padding=(3, 3),
         ceil_mode=True,
         count_include_pad=False,
         test_name='ceil mode with padding')

    test(
        input_channels=3, in_height=6, in_width=6, stride=(4, 4),
        padding=(1, 1),
        kernel_height=2, kernel_width=2, ceil_mode=True,
        ceil_mode_note_height=True, ceil_mode_note_width=True,
        count_include_pad=False,
        test_name='ceil mode with note, https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html')

    test(
        input_channels=3, in_height=6, in_width=40, stride=(4, 5),
        padding=(1, 1),
        kernel_height=2, kernel_width=5, ceil_mode=True,
        ceil_mode_note_height=True, ceil_mode_note_width=False,
        count_include_pad=False,
        test_name='ceil mode with note for height, https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html')

    test(
        input_channels=3, in_height=40, in_width=6, stride=(5, 4),
        padding=(1, 1),
        kernel_height=5, kernel_width=2, ceil_mode=True,
        ceil_mode_note_height=False, ceil_mode_note_width=True,
        count_include_pad=False,
        test_name='ceil mode with note for width, https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         padding=(3, 2),
         count_include_pad=True,
         test_name='count include pad')

    test(input_channels=3,
         in_height=40,
         in_width=6,
         stride=(5, 4),
         padding=(1, 1),
         kernel_height=5,
         kernel_width=2,
         ceil_mode=True,
         count_include_pad=True,
         test_name='ceil mode with note for width and count include pad')

    test(input_channels=3,
         in_height=48,
         in_width=52,
         stride=(1, 2),
         padding=(2, 2),
         kernel_height=5,
         kernel_width=5,
         divisor_override=3,
         test_name='divisor override')


if __name__ == '__main__':
    main()
