from typing import Sequence
import torch
import ai3
import torch.nn.functional as F

CONV = 'conv'
MHA = 'mha'


def get_grad(model, input, target):
    out = model(input) if isinstance(input, torch.Tensor) else model(*input)
    loss = F.mse_loss(out, target)
    model.zero_grad()
    loss.backward(retain_graph=True)
    return {name: param.grad.clone() for name, param in model.named_parameters()}


def test_with(input, model, op, mes):
    if isinstance(input, torch.Tensor):
        input.requires_grad = True
    else:
        for t in input:
            t.requires_grad = True
    out = model(input) if isinstance(input, torch.Tensor) else model(*input)

    target = torch.randn(out.shape)

    grad_torch = get_grad(model, input, target)
    if op == CONV:
        ai3.swap_conv2d(model)
    elif op == MHA:
        ai3.swap_mha(model)
    grad_ai3 = get_grad(model, input, target)

    same_gradients = True
    for name in grad_torch:  # TODO after this works put it in the get_grad function
        if op == MHA:
            if name == 'attn.in_proj_weight':
                grad_ai3['attn.in_proj_weight'] = torch.cat([
                    grad_ai3['attn.q_proj_weight'],
                    grad_ai3['attn.k_proj_weight'],
                    grad_ai3['attn.v_proj_weight'],
                ], dim=0)
            elif name == 'attn.in_proj_bias':
                grad_ai3['attn.in_proj_bias'] = torch.cat([
                    grad_ai3['attn.bias_q_in'],
                    grad_ai3['attn.bias_k_in'],
                    grad_ai3['attn.bias_v_in'],
                ], dim=0)
            grad_ai3['attn.out_proj.weight'] = grad_ai3['attn.out_proj_weight']
            grad_ai3['attn.out_proj.bias'] = grad_ai3['attn.out_proj_bias']

        if not torch.allclose(grad_torch[name], grad_ai3[name]):
            print(
                f'Gradients for {name} on {mes} differ')
            print('first 10 torch:', ' '.join(
                map(str, grad_torch[name].flatten()[:10].tolist())))
            print('first 10 ai3:', ' '.join(
                map(str, grad_ai3[name].flatten()[:10].tolist())))
            same_gradients = False

    if same_gradients:
        print(
            f'Gradients are the same for {op} on {mes}')
    else:
        print(
            f'Gradients are different for {op} on {mes}')


def conv2d():
    class ConvModel(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super(ConvModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size)

        def forward(self, x):
            x = self.conv1(x)
            return x
    test_with(torch.randn(
        10, 300, 300), ConvModel(10, 32, 3), CONV, 'no batch')
    test_with(torch.randn(
        1, 3, 224, 224), ConvModel(3, 16, (4, 3)), CONV, 'batch = 1')
    test_with(torch.randn(
        10, 10, 512, 52), ConvModel(10, 5, 5), CONV, 'batch = 10')


def mha():
    class MHAModel(torch.nn.Module):
        def __init__(
                self, embed_dim=512, num_heads=8, kdim=None, vdim=None,
                bias=True, add_bias_kv=False, batch_first=True,
                add_zero_attn=False, dtype=None):
            super(MHAModel, self).__init__()
            self.attn = torch.nn.MultiheadAttention(
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

        def forward(self, q, k, v):
            return self.attn(q, k, v, need_weights=False)[0]

    test_with(
        (torch.randn(10, 50, 512),
         torch.randn(10, 50, 512),
         torch.randn(10, 50, 512)),
        MHAModel(512),
        MHA, 'basic')
    test_with(
        (torch.randn(50, 10, 512),
         torch.randn(50, 10, 512),
         torch.randn(50, 10, 512)),
        MHAModel(512, batch_first=False),
        MHA, 'basic no batch first')
    test_with(
        (torch.randn(10, 50, 300),
         torch.randn(10, 50, 200),
         torch.randn(10, 50, 150)),
        MHAModel(300, 5, 200, 150),
        MHA, 'different kdim and vdim')
    # TODO might be possible to support gradients for the data if we don't project?
    # TODO do attn_mask and causal
    test_with(
        (torch.randn(10, 60, 80),
         torch.randn(10, 60, 80),
         torch.randn(10, 60, 80)),
        MHAModel(80, 5, add_bias_kv=True, add_zero_attn=False),
        MHA, 'with bias_kv')
