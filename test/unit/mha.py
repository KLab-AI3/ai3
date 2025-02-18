import torch
from torch import nn
import ai3
from test import compare_tensors


class MHA(nn.Module):
    def __init__(
            self, embed_dim, num_heads, kdim, vdim, bias, add_bias_kv,
            batch_first, add_zero_attn, dtype):
        super(MHA, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, kdim=kdim, vdim=vdim, bias=bias,
            add_bias_kv=add_bias_kv, batch_first=batch_first,
            add_zero_attn=add_zero_attn, dtype=dtype)

    def forward(self, q, k, v):
        attn_output, _ = self.attn(q, k, v, need_weights=False)
        return attn_output


def test(*, num_samples, seq_len_q: int, embed_dim: int, num_heads: int,
         kdim=None, vdim=None, seq_len_k=None,
         bias: bool = False, add_bias_kv=False, batch_first=False,
         add_zero_attn: bool = False, use_same=False, test_name: str) -> None:
    if use_same:
        assert (kdim or vdim) is None
    kdim = kdim or embed_dim
    vdim = vdim or embed_dim
    seq_len_k = seq_len_k or seq_len_q
    dtype = torch.float32
    assert kdim and vdim
    if num_samples is not None:
        if batch_first:
            query = torch.randn(
                (num_samples, seq_len_q, embed_dim), dtype=dtype)
            key = torch.randn(
                (num_samples, seq_len_k, kdim), dtype=dtype)
            value = torch.randn(
                (num_samples, seq_len_k, vdim), dtype=dtype)
        else:
            query = torch.randn(
                (seq_len_q, num_samples, embed_dim), dtype=dtype)
            key = torch.randn(
                (seq_len_k, num_samples, kdim), dtype=dtype)
            value = torch.randn(
                (seq_len_k, num_samples, vdim), dtype=dtype)
    else:
        query = torch.randn(
            (seq_len_q, embed_dim), dtype=dtype)
        key = torch.randn(
            (seq_len_k, kdim), dtype=dtype)
        value = torch.randn(
            (seq_len_k, vdim), dtype=dtype)

    orig = MHA(embed_dim, num_heads, kdim, vdim, bias,
               add_bias_kv, batch_first, add_zero_attn, dtype)
    inputs = (query, key, value) if not use_same else (query, query, query)

    torch_output = orig(*inputs)

    ai3.swap_mha(orig)
    ai3_output = orig(*inputs)
    compare_tensors(
        ai3_output, torch_output, test_name, print_diff=False, print_same=True, atol=1e-3)


def main():
    print('MHA')
    test(num_samples=20,
         seq_len_q=10,
         embed_dim=64,
         num_heads=4,
         bias=False,
         batch_first=True,
         test_name='batched batch_first')

    test(num_samples=None,
         seq_len_q=20,
         embed_dim=64,
         num_heads=4,
         bias=True,
         batch_first=True,
         test_name='not batched bias')

    test(num_samples=None,
         seq_len_q=10,
         embed_dim=64,
         num_heads=1,
         bias=True,
         batch_first=False,
         test_name='one head not batched no batch_first')

    test(num_samples=20,
         seq_len_q=20,
         embed_dim=64,
         num_heads=4,
         bias=True,
         batch_first=False,
         test_name='batched no batch_first')

    test(num_samples=20,
         seq_len_q=50,
         embed_dim=128,
         num_heads=16,
         bias=True,
         batch_first=True,
         test_name='different dimensions')

    test(num_samples=20,
         seq_len_q=50,
         seq_len_k=25,
         embed_dim=64,
         num_heads=16,
         kdim=32,
         vdim=16,
         bias=True,
         test_name='unique embed, kdim, vdim, seq_len_q, seq_len_k')

    test(num_samples=20,
         seq_len_q=50,
         embed_dim=64,
         num_heads=4,
         bias=True,
         use_same=True,
         test_name='using same tensor')

    test(num_samples=20,
         seq_len_q=1000,
         embed_dim=400,
         num_heads=8,
         bias=True,
         test_name='seq_len_q=1000')

    test(num_samples=None, seq_len_q=50, embed_dim=64, num_heads=4, kdim=32,
         vdim=16, bias=False, add_bias_kv=True, batch_first=True,
         test_name='not batched different unique embed, k, v, with add_bias_kv')

    test(num_samples=None, seq_len_q=50, embed_dim=64, num_heads=4, kdim=32,
         vdim=16, bias=False, add_bias_kv=True, batch_first=False,
         test_name='not batched different unique embed, k, v, with add_bias_kv')

    test(num_samples=3,
         seq_len_q=40,
         embed_dim=48,
         num_heads=4,
         kdim=20,
         vdim=16,
         bias=True,
         add_bias_kv=True,
         add_zero_attn=True,
         test_name='batched, unique, bias, add_bias_kv, add_zero_attn')

    test(num_samples=None,
         seq_len_q=40,
         embed_dim=48,
         num_heads=4,
         bias=False,
         add_bias_kv=True,
         add_zero_attn=False,
         test_name='not unique, add_bias_kv')


if __name__ == '__main__':
    main()
