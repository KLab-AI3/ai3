import torch
from torch import nn
import ai3
from test import compare_tensors


class Flatten(nn.Module):
    def __init__(self, start_dim, end_dim):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return torch.flatten(input, self.start_dim, self.end_dim)


def test(*, in_shape,
         start_dim: int = 0,
         end_dim: int = -1,
         test_name: str) -> None:
    input = torch.randn(
        in_shape, dtype=torch.float32)

    orig = Flatten(start_dim, end_dim)

    torch_output = orig(input)

    model = ai3.convert(orig)
    ai3_output = model.predict(input)
    compare_tensors(
        ai3_output, torch_output, test_name)


def main():
    print('FLATTEN')
    test(in_shape=(2, 2, 2), test_name='small basic')
    test(in_shape=(2, 2, 2), start_dim=1,
         test_name='small from 1')
    test(in_shape=(2, 2, 2), start_dim=2,
         test_name='small from 2')
    test(in_shape=(3, 7, 5), start_dim=0,
         end_dim=1, test_name='small 0 to 1')
    test(in_shape=(3, 7, 9), start_dim=0,
         end_dim=2, test_name='small 0 to 2')
    test(in_shape=(8, 6, 3), start_dim=1,
         end_dim=2, test_name='small 1 to 2')
    test(in_shape=(8, 6, 3), start_dim=1,
         end_dim=1, test_name='start = end')
    test(in_shape=(1, 2, 3, 4, 5), start_dim=2,
         end_dim=4, test_name='5 dim 2 to 4')
    test(in_shape=(4, 7, 1, 4, 3, 2, 3,
         7, 9, 1), test_name='10 dim')
    test(in_shape=(4, 7, 1, 4, 3, 2, 3, 7, 9, 1),
         start_dim=5, end_dim=7, test_name='10 dim 5 to 7')


if __name__ == '__main__':
    main()
