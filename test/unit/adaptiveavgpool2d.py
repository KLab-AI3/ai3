import torch
from torch import nn
import ai3
from test import compare_tensors
from typing import Optional


class AdaptiveAvgPool2D(nn.Module):
    def __init__(self, output_shape):
        super(AdaptiveAvgPool2D, self).__init__()
        self.adapavgpool2d = nn.AdaptiveAvgPool2d(output_shape)

    def forward(self, input):
        return self.adapavgpool2d(input)


def test(
    *, num_samples: Optional[int] = None, input_channels: int,
        in_height: int, in_width: int, output_shape, test_name: str) -> None:
    if num_samples is None:
        input = torch.randn(input_channels, in_height,
                            in_width, dtype=torch.float32)
    else:
        input = torch.randn(num_samples, input_channels, in_height,
                            in_width, dtype=torch.float32)

    orig = AdaptiveAvgPool2D(output_shape)
    torch_output = orig(input)
    model = ai3.convert(orig)
    ai3_output = model.predict(input)
    compare_tensors(
        ai3_output, torch_output, test_name)


def main():
    print('ADAPTIVE AVG POOL 2D')

    test(input_channels=3,
         in_height=30,
         in_width=30,
         output_shape=(6, 6),
         test_name='out is multiple of in')
    test(num_samples=3,
         input_channels=3,
         in_height=40,
         in_width=30,
         output_shape=(4, 3),
         test_name='separate multiples')
    test(input_channels=10,
         in_height=15,
         in_width=15,
         output_shape=(None, 3),
         test_name='first None')
    test(num_samples=5,
         input_channels=10,
         in_height=15,
         in_width=15,
         output_shape=(5, None),
         test_name='second None')
    test(num_samples=10,
         input_channels=10,
         in_height=50,
         in_width=50,
         output_shape=(None, None),
         test_name='both None')

if __name__ == '__main__':
    main()
