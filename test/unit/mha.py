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


def test(*, num_samples, seq_len: int, embed_dim: int, num_heads: int,
         kdim=None, vdim=None,
         bias: bool = False, add_bias_kv=False, batch_first=False,
         add_zero_attn: bool = False, test_name: str) -> None:
    kdim = kdim or embed_dim
    vdim = vdim or embed_dim
    dtype = torch.float32
    assert kdim and vdim
    if num_samples is not None:
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
    else:
        query = torch.randn(
                (seq_len, embed_dim), dtype=dtype)
        key = torch.randn(
                (seq_len, kdim), dtype=dtype)
        value = torch.randn(
                (seq_len, vdim), dtype=dtype)

    orig = MHA(embed_dim, num_heads, kdim, vdim, bias,
               add_bias_kv, batch_first, add_zero_attn, dtype)
    torch_output = orig(query, key, value)

    ai3.swap_mha(orig)
    ai3_output = orig(query, key, value)
    compare_tensors(
        ai3_output, torch_output, test_name, print_diff=False, atol=1e-3)


def main():
    print('MHA')

    test(num_samples=20,
         seq_len=10,
         embed_dim=64,
         num_heads=4,
         bias=True,
         batch_first=True,
         test_name='batched batch_first')

    test(num_samples=None,
         seq_len=20,
         embed_dim=64,
         num_heads=4,
         bias=True,
         batch_first=True,
         test_name='not batched bias')

    test(num_samples=None,
         seq_len=10,
         embed_dim=64,
         num_heads=1,
         bias=False,
         batch_first=False,
         test_name='not batched no bias no batch_first')

    test(num_samples=20,
         seq_len=20,
         embed_dim=64,
         num_heads=4,
         bias=True,
         batch_first=False,
         test_name='batched no batch_first')
    exit(0)

    test(num_samples=20,
         seq_len=50,
         embed_dim=128,
         num_heads=16,
         bias=True,
         batch_first=True,
         test_name='different dimensions')

    # TODO fail on 192, CUDNN_STATUS_BAD_PARAM
    # 2220 I! CuDNN (v90600 74) function cudnnSetAttnDescriptor() called:
    #     attnMode: type=unsigned; val=CUDNN_ATTN_QUERYMAP_ALL_TO_ONE|CUDNN_ATTN_DISABLE_PROJ_BIASES (0x0);
    #     nHeads: type=int; val=16;
    #     smScaler: type=double; val=1;
    #     dataType: type=cudnnDataType_t; val=CUDNN_DATA_FLOAT (0);
    #     computePrec: type=cudnnDataType_t; val=CUDNN_DATA_FLOAT (0);
    #     mathType: type=cudnnMathType_t; val=CUDNN_DEFAULT_MATH (0);
    #     attnDropoutDesc: type=cudnnDropoutDescriptor_t; val=NULL_PTR;
    #     postDropoutDesc: type=cudnnDropoutDescriptor_t; val=NULL_PTR;
    #     qSize: type=int; val=64;
    #     kSize: type=int; val=32;
    #     vSize: type=int; val=16;
    #     qProjSize: type=int; val=4;
    #     kProjSize: type=int; val=2;
    #     vProjSize: type=int; val=1;
    #     oProjSize: type=int; val=16;
    #     qoMaxSeqLength: type=int; val=50;
    #     kvMaxSeqLength: type=int; val=50;
    #     maxBatchSize: type=int; val=1;
    #     maxBeamSize: type=int; val=1;
    # Time: 2025-02-04T14:45:55.407597 (0d+0h+0m+1s since start)
    # Process=3975292; Thread=3975292; GPU=NULL; Handle=NULL; StreamId=NULL.
    test(num_samples=None,
         seq_len=50,
         embed_dim=64,
         num_heads=16,
         kdim=32,
         vdim=16,
         test_name='not batched different unique embed, k, v')
    exit(0)
    test(num_samples=None, seq_len=50, embed_dim=64, num_heads=4, kdim=32,
         vdim=16, bias=False, add_bias_kv=True,
         test_name='not batched different unique embed, k, v, with add_bias_kv')

    test(num_samples=3,
         seq_len=40,
         embed_dim=48,
         num_heads=4,
         kdim=20,
         vdim=16,
         bias=True,
         add_bias_kv=True,
         add_zero_attn=True,
         test_name='batched, unique, bias, add_bias_kv, add_zero_attn')
    test(num_samples=None,
         seq_len=40,
         embed_dim=48,
         num_heads=4,
         bias=False,
         add_bias_kv=True,
         add_zero_attn=False,
         test_name='not unique, add_bias_kv')
    test(num_samples=3,
         seq_len=40,
         embed_dim=48,
         num_heads=4,
         kdim=20,
         vdim=16,
         bias=True,
         add_bias_kv=False,
         test_name='unique, with bias')


if __name__ == '__main__':
    main()
