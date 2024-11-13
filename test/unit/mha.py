import torch
from torch import nn
import ai3
from test import compare_tensors


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim, vdim, bias, add_bias_kv, batch_first, add_zero_attn):
        super(MHA, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, kdim=kdim, vdim=vdim,
                                          bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first,
                                          add_zero_attn=add_zero_attn)

    def forward(self, q, k, v):
        attn_output, _ = self.attn(q, k, v, need_weights=False)
        return attn_output


def test(*, num_samples, seq_len: int, embed_dim: int, num_heads: int,
         kdim = None, vdim = None ,
         bias: bool = False, add_bias_kv = False, batch_first = None,
         add_zero_attn: bool = False, test_name: str) -> None:
    kdim = kdim or embed_dim
    vdim = vdim or embed_dim
    assert kdim and vdim
    if num_samples:
        input = torch.randn(
            (num_samples, seq_len, embed_dim), dtype=torch.float32)
        key = torch.randn(
            (num_samples, seq_len, kdim), dtype=torch.float32)
        value = torch.randn(
            (num_samples, seq_len, vdim), dtype=torch.float32)

    else:
        input = torch.randn(
            seq_len, embed_dim, dtype=torch.float32)
        key = torch.randn(
            (seq_len, kdim), dtype=torch.float32)
        value = torch.randn(
            (seq_len, vdim), dtype=torch.float32)

    orig = MHA(embed_dim, num_heads, kdim, vdim, bias, add_bias_kv, batch_first, add_zero_attn)
    torch_output = orig(input, key, value)

    ai3.swap_mha(orig)
    ai3_output = orig(input, key, value)
    compare_tensors(
        ai3_output, torch_output, test_name, print_diff=False)


# TODO check other hparams
print('MHA')
test(num_samples=None,
     seq_len=10,
     embed_dim=64,
     num_heads=4,
     test_name='not batched no bias')
test(num_samples=None,
     seq_len=20,
     embed_dim=64,
     num_heads=4,
     bias=True,
     test_name='not batched bias')
test(num_samples=20,
     seq_len=10,
     embed_dim=64,
     num_heads=4,
     bias=True,
     batch_first=True,
     test_name='batched batch_first')
test(num_samples=20,
     seq_len=50,
     embed_dim=128,
     num_heads=16,
     bias=True,
     batch_first=False,
     test_name='batched no batch_first')
test(num_samples=None,
     seq_len=50,
     embed_dim=64,
     num_heads=16,
     kdim=32,
     vdim=16,
     test_name='not batched different unique embed, k, v')
test(num_samples=None,
     seq_len=50,
     embed_dim=64,
     num_heads=4,
     kdim=32,
     vdim=16,
     bias=False,
     add_bias_kv=True,
     test_name='not batched different unique embed, k, v, with add_bias_kv')
test(num_samples=3,
     seq_len=40,
     embed_dim=48,
     num_heads=4,
     kdim=20,
     vdim=16,
     bias=False,
     add_bias_kv=True,
     add_zero_attn=True,
     test_name='batched, unique, bias, add_bias_kv, add_zero_attn')

