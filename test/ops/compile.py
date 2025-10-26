import torch
from torch import nn
import ai3
import platform
from test import compare_tensors

PASS_MES = 'ai3 and torch Models compiled with torch.compile produce same outputs '


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1,
            bias=False)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        return x


def compile(orig):
    if platform.system() == 'Darwin':
        return torch.compile(orig, backend='aot_eager')
    else:
        return torch.compile(orig)


def conv2d():
    input_data = torch.randn(3, 224, 224)
    orig = ConvNet()
    tar = orig(input_data)

    ai3.swap_conv2d(orig)
    swap_comped = compile(orig)
    swap_comped_out = swap_comped(input_data)

    assert torch.allclose(
        swap_comped_out, tar, atol=1e-6)
    print(PASS_MES + 'conv2d')


class MHA(nn.Module):
    def __init__(
            self, embed_dim=512, num_heads=8, kdim=None, vdim=None, bias=True,
            add_bias_kv=False, batch_first=True, add_zero_attn=False,
            dtype=None):
        super(MHA, self).__init__()
        self.attn1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim or embed_dim,
            vdim=vdim or embed_dim,
            bias=bias,
            add_bias_kv=add_bias_kv,
            batch_first=batch_first,
            add_zero_attn=add_zero_attn,
            dtype=dtype
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim or embed_dim,
            vdim=vdim or embed_dim,
            bias=bias,
            add_bias_kv=add_bias_kv,
            batch_first=batch_first,
            add_zero_attn=add_zero_attn,
            dtype=dtype
        )

        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_output1, _ = self.attn1(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
            need_weights=False)
        x = self.norm1(x + attn_output1)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        attn_output2, _ = self.attn2(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
            need_weights=False)
        x = self.norm3(x + attn_output2)
        return x


def mha():
    input_data = torch.randn(2, 10, 512)

    orig = MHA(embed_dim=512, num_heads=8)
    orig.eval()
    tar = orig(input_data)

    ai3.swap_mha(orig)
    swap_comped = compile(orig)
    swap_comped_out = swap_comped(input_data)

    compare_tensors(swap_comped_out, tar)
    assert torch.allclose(
        swap_comped_out, tar, atol=1e-6)
    print(PASS_MES + 'mha')
