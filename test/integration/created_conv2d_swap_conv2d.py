import torch
import ai3
from test import compare_tensors
from example.simple_conv2d import SimpleConvNet


def _run(data, mes):
    model = SimpleConvNet()
    target = model(data)
    ai3.swap_conv2d(model)
    output = model(data)
    compare_tensors(output, target.detach().numpy(), mes)


def run():
    print("SIMPLE CREATED CONV NET SWAPPING CONV2D")
    _run(torch.randn(3, 224, 224), "simple conv2d just conv2d")
    _run(torch.randn(5, 3, 224, 224), "simple conv2d multi sample just conv2d")


if __name__ == "__main__":
    run()
