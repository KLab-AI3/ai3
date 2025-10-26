import torch
from torch import ops  # type: ignore
from ai3 import _core

PASS_MES = 'Passed opcheck for '


def conv2d():
    input_size = (1, 3, 224, 224)
    kernel_size = (3, 3, 5, 5)
    samples = [(
        torch.randn(
            input_size, requires_grad=grad),
        torch.randn(
            kernel_size, requires_grad=grad),
        torch.randn(
            kernel_size[0], requires_grad=grad),
        1, 1, 1, 1, 1, 1, 0, 1, 'default') for grad in [False, True]]

    assert callable(ops.ai3.conv2d)
    for samp in samples:
        torch.library.opcheck(ops.ai3.conv2d, samp)  # type: ignore
    print(PASS_MES + 'conv2d')


def mha():
    batch_size, seq_len, embed_dim, num_heads = 2, 16, 64, 8
    head_dim = embed_dim // num_heads
    k_dim, v_dim = head_dim, head_dim

    assert callable(ops.ai3.mha)
    assert callable(ops.ai3.mha_backward)

    for grad in [False, True]:
        for mem_format in [_core.MHAMemFormat.NSE, _core.MHAMemFormat.SNE]:

            first_two = (batch_size, seq_len) if mem_format == _core.MHAMemFormat.NSE else (
                seq_len, batch_size)

            query = torch.randn(*first_two, embed_dim, requires_grad=grad)
            key = torch.randn(*first_two, k_dim, requires_grad=grad)
            value = torch.randn(*first_two, v_dim, requires_grad=grad)
            q_proj = torch.randn(embed_dim, embed_dim, requires_grad=grad)
            k_proj = torch.randn(embed_dim, k_dim, requires_grad=grad)
            v_proj = torch.randn(embed_dim, v_dim, requires_grad=grad)
            out_proj = torch.randn(embed_dim, embed_dim, requires_grad=grad)
            q_proj_bias = torch.randn(embed_dim, requires_grad=grad)
            k_proj_bias = torch.randn(embed_dim, requires_grad=grad)
            v_proj_bias = torch.randn(embed_dim, requires_grad=grad)
            out_proj_bias = torch.randn(embed_dim, requires_grad=grad)

            args = (
                query, key, value,
                q_proj, k_proj, v_proj, out_proj,
                q_proj_bias, k_proj_bias, v_proj_bias, out_proj_bias,
                mem_format,
                None, None,
                False,
                num_heads,
                k_dim, v_dim, embed_dim,
                0,
                None,
                False,
                None,
                True,
                False,
                True,
                "default"
            )

            torch.library.opcheck(ops.ai3.mha, args=args)  # type: ignore

    print(PASS_MES + "mha")
