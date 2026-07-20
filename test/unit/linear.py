import torch
from torch import nn
import ai3
from test import compare_tensors


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input):
        return self.linear(input)


def test(*, num_samples, in_features: int, out_features: int,
         with_bias: bool = False,
         test_name: str) -> None:
    if num_samples:
        input = torch.randn(
            (num_samples, in_features), dtype=torch.float32)
    else:
        input = torch.randn(
            in_features, dtype=torch.float32)

    orig = Linear(in_features, out_features, with_bias)
    torch_output = orig(input)

    model = ai3.convert(orig)
    ai3_output = model.predict(input, torch.Tensor)
    compare_tensors(
        ai3_output, torch_output, test_name)


def main():
    print('LINEAR')
    test(num_samples=None,
         in_features=2,
         out_features=2,
         test_name='square')
    test(num_samples=None,
         in_features=4,
         out_features=4,
         with_bias=True,
         test_name='square bias')
    test(num_samples=None,
         in_features=100,
         out_features=5,
         test_name='in > out')
    test(num_samples=None,
         in_features=5,
         out_features=100,
         test_name='out > in')
    test(num_samples=None,
         in_features=40,
         out_features=30,
         with_bias=True,
         test_name='10s with bias')
    test(num_samples=None,
         in_features=348,
         out_features=498,
         with_bias=True,
         test_name='100s with bias')
    test(num_samples=5,
         in_features=300,
         out_features=400,
         with_bias=False,
         test_name='100s no bias multiple samples')
    test(num_samples=5,
         in_features=348,
         out_features=498,
         with_bias=True,
         test_name='100s with bias multiple samples')


if __name__ == '__main__':
    main()
