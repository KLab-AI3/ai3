from . import backward, compile, opcheck, train
from ai3 import swap_torch # to initialize the torch.ops.ai3
_ = swap_torch

def run():
    opcheck.conv2d()
    compile.conv2d()
    backward.conv2d()
    train.conv2d()
