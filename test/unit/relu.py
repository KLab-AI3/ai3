import torch
import ai3
from torch import nn
from test import compare_tensors


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        return torch.relu(input)


def test(*, input_shape,
         test_name: str) -> None:
    input = torch.randn(
        input_shape, dtype=torch.float32)
    orig = ReLU()
    torch_output = orig(input)
    model = ai3.convert(orig)
    ai3_output = model.predict(input)
    compare_tensors(
        ai3_output, torch_output, test_name)


def main():
    print('RELU')
    test(input_shape=1,
         test_name='one')
    test(input_shape=(1, 4, 56, 48),
         test_name='normal')
    shape = (3, 1, 4, 5, 6, 8, 1, 1, 8, 4)
    test(input_shape=shape,
         test_name=f'{len(shape)} dim')

if __name__ == '__main__':
    main()
