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

    assert (callable(ops.ai3.conv2d))
    for samp in samples:
        torch.library.opcheck(
            ops.ai3.conv2d, samp)
    print(PASS_MES + 'conv2d')


def mha():
    batch_size, seq_len, embed_dim, num_heads = 2, 16, 64, 8
    head_dim = embed_dim // num_heads
    k_dim, v_dim = head_dim, head_dim

    samples = [
        {'query': torch.randn(
            batch_size, seq_len, embed_dim, requires_grad=grad),
         'key': torch.randn(
             batch_size, seq_len, k_dim, requires_grad=grad),
         'value': torch.randn(
             batch_size, seq_len, v_dim, requires_grad=grad),
         'q_proj': torch.randn(embed_dim, embed_dim, requires_grad=grad),
         'k_proj': torch.randn(embed_dim, embed_dim, requires_grad=grad),
         'v_proj': torch.randn(embed_dim, embed_dim, requires_grad=grad),
         'out_proj': torch.randn(
             embed_dim, embed_dim, requires_grad=grad),
         'q_proj_bias': torch.randn(embed_dim, requires_grad=grad),
         'k_proj_bias': torch.randn(embed_dim, requires_grad=grad),
         'v_proj_bias': torch.randn(embed_dim, requires_grad=grad),
         'out_proj_bias': torch.randn(embed_dim, requires_grad=grad),
         'mem_fmt': _core.MHAMemFormat.NSE, 'k_bias': None, 'v_bias': None,
         'add_zero_attn': False, 'num_heads': num_heads, 'k_dim': k_dim,
         'v_dim': v_dim, 'embed_dim': embed_dim, 'dropout': 0,
         'key_padding_mask': None, 'need_weights': False, 'attn_mask': None,
         'average_attn_weights': True, 'is_causal': False,
         'need_to_project': True, 'algorithm': 'default'}
        for grad in [False, True]]

    assert callable(ops.ai3.mha)
    for samp in samples:
        torch.library.opcheck(ops.ai3.mha, (), kwargs=samp)
        print('--------')
    print(PASS_MES + 'mha')
