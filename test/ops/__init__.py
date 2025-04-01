from . import backward, compile, opcheck, train
from ai3 import swap_torch  # to initialize the torch.ops.ai3
_ = swap_torch


def run_conv2d():
    for m in [opcheck, compile, backward, train]:
        m.conv2d()


def run_mha():
    for m in [opcheck, compile, backward, train]:
        m.mha()


def run():
    run_conv2d()
    run_mha()
