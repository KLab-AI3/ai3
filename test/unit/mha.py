import torch
from torch import nn
import ai3
from test import compare_tensors


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, batch_first):
        super(MHA, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=batch_first)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output


def test(*, num_samples, seq_len: int, embed_dim: int, num_heads: int,
         bias: bool = False, batch_first = None,
         test_name: str) -> None:
    if num_samples:
        input = torch.randn(
            (num_samples, seq_len, embed_dim), dtype=torch.float32)
    else:
        input = torch.randn(
            seq_len, embed_dim, dtype=torch.float32)

    orig = MHA(embed_dim, num_heads, bias, batch_first)
    torch_output = orig(input)

    ai3.swap_mha(orig)
    ai3_output = orig(input)
    compare_tensors(
        ai3_output, torch_output, test_name)


# TODO check other hparams, add_bias_kv first
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
