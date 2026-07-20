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


class ConvBwd(nn.Module):
    def __init__(self):
        super(ConvBwd, self).__init__()
        self.conv = nn.Conv2d(3, 8, 3)

    def forward(self, x):
        return self.conv(x)


def compile(orig):
    if platform.system() == 'Darwin':
        return torch.compile(orig, backend='aot_eager')
    else:
        return torch.compile(orig)


def backward_grads(run, module, inputs):
    for t in inputs:
        t.grad = None
    module.zero_grad()
    run(*inputs).sum().backward()
    return ([t.grad.clone() for t in inputs],
            {name: p.grad.clone() for name, p in module.named_parameters()})


def conv2d():
    input_data = torch.randn(3, 224, 224)
    orig = ConvNet()
    tar = orig(input_data)

    ai3.swap_conv2d(orig)
    swap_comped = compile(orig)
    swap_comped_out = swap_comped(input_data)

    compare_tensors(swap_comped_out, tar, PASS_MES + 'conv2d',
                    print_diff=False)

    bwd = ConvBwd()
    bwd.eval()
    inputs = (torch.randn(2, 3, 16, 16, requires_grad=True),)
    tar_input_grads, tar_param_grads = backward_grads(bwd, bwd, inputs)

    ai3.swap_conv2d(bwd)
    bwd_comped = compile(bwd)
    swap_input_grads, swap_param_grads = backward_grads(
        bwd_comped, bwd, inputs)

    for name, tar_grad, swap_grad in zip(('dinput',),
                                         tar_input_grads, swap_input_grads):
        compare_tensors(swap_grad, tar_grad,
                        f'{PASS_MES}conv2d backward {name}',
                        print_diff=False)
    for name in tar_param_grads:
        compare_tensors(swap_param_grads[name], tar_param_grads[name],
                        f'{PASS_MES}conv2d backward {name}',
                        print_diff=False)


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


class MHASingle(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super(MHASingle, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)

    def forward(self, q, k, v):
        return self.attn(q, k, v, need_weights=False)[0]


def mha():
    input_data = torch.randn(2, 10, 512)

    orig = MHA(embed_dim=512, num_heads=8)
    orig.eval()
    tar = orig(input_data)

    ai3.swap_mha(orig)
    swap_comped = compile(orig)
    swap_comped_out = swap_comped(input_data)

    compare_tensors(swap_comped_out, tar, PASS_MES + 'mha',
                    print_diff=False)

    bwd = MHASingle()
    bwd.eval()
    inputs = tuple(torch.randn(2, 10, 64, requires_grad=True)
                   for _ in range(3))
    tar_input_grads, _ = backward_grads(bwd, bwd, inputs)

    ai3.swap_mha(bwd)
    bwd_comped = compile(bwd)
    swap_input_grads, _ = backward_grads(bwd_comped, bwd, inputs)

    for name, tar_grad, swap_grad in zip(('dq', 'dk', 'dv'),
                                         tar_input_grads, swap_input_grads):
        compare_tensors(swap_grad, tar_grad,
                        f'{PASS_MES}mha backward {name}',
                        print_diff=False)
