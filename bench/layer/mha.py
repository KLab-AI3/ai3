import copy
import torch
from bench import predict_show_time
from torch import nn
import ai3
from test import compare_tensors
from copy import deepcopy

N = 100

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


def run_on(*, num_samples: int, seq_len: int, embed_dim: int, num_heads: int,
         kdim=None, vdim=None,
         bias: bool = False, add_bias_kv=False, batch_first=False,
         add_zero_attn: bool = False, use_same=False,
         dtype=torch.float32) -> None:
    if use_same:
        assert (kdim or vdim) is None
    kdim = kdim or embed_dim
    vdim = vdim or embed_dim
    dtype = dtype
    if batch_first:
        query = torch.randn(
            (num_samples, seq_len, embed_dim), dtype=dtype)
        key = torch.randn(
            (num_samples, seq_len, kdim), dtype=dtype)
        value = torch.randn(
            (num_samples, seq_len, vdim), dtype=dtype)
    else:
        query = torch.randn(
            (seq_len, num_samples, embed_dim), dtype=dtype)
        key = torch.randn(
            (seq_len, num_samples, kdim), dtype=dtype)
        value = torch.randn(
            (seq_len, num_samples, vdim), dtype=dtype)

    orig = MHA(embed_dim, num_heads, kdim, vdim, bias,
               add_bias_kv, batch_first, add_zero_attn, dtype)
    inputs = (query, key, value) if not use_same else (query, query, query)
    orig_cpy = copy.deepcopy(orig)
    ai3.swap_mha(orig)
    out = predict_show_time(orig, inputs, f'ai3: '
                    f'num_heads: {num_heads}, bias: {bias},'
                    f'q: {num_samples},{seq_len},{embed_dim}, '
                    f'k: {num_samples},{seq_len},{kdim}, '
                    f'v: {num_samples},{seq_len},{kdim}')
    orig_out = predict_show_time(
            orig_cpy, inputs, f'pytorch: '
                    f'num_heads: {num_heads}, bias: {bias}, '
                    f'q: {num_samples},{seq_len},{embed_dim}, '
                    f'k: {num_samples},{seq_len},{kdim}, '
                    f'v: {num_samples},{seq_len},{kdim}')
    out = predict_show_time(orig, inputs, f'ai3:'
                    f'num_heads: {num_heads}, bias: {bias}, '
                    f'q: {num_samples},{seq_len},{embed_dim}, '
                    f'k: {num_samples},{seq_len},{kdim}, '
                    f'v: {num_samples},{seq_len},{kdim}')

    compare_tensors(out, orig_out, print_pass=False, print_diff=True, atol=1e-3)

print('Multihead Attention')
run_on(num_samples=N, seq_len=100, embed_dim=64, num_heads=4, bias=True)
print('-------------')
run_on(num_samples=N, seq_len=2000, embed_dim=64, num_heads=4, bias=True)
