import torch
import ai3
import torch.nn.functional as F

CONV = 'conv'
MHA = 'mha'


def as_tuple(input):
    return (input,) if isinstance(input, torch.Tensor) else tuple(input)


def get_grad(model, inputs, target):
    for t in inputs:
        t.grad = None
    out = model(*inputs)
    loss = F.mse_loss(out, target)
    model.zero_grad()
    loss.backward(retain_graph=True)
    param_grads = {name: param.grad.clone()
                   for name, param in model.named_parameters()}
    input_grads = [t.grad.clone() if t.grad is not None else None
                   for t in inputs]
    return param_grads, input_grads


def remap_mha_ai3(grad_ai3, torch_names):
    g = dict(grad_ai3)
    if 'attn.in_proj_weight' in torch_names and 'attn.q_proj_weight' in g:
        g['attn.in_proj_weight'] = torch.cat([
            g['attn.q_proj_weight'],
            g['attn.k_proj_weight'],
            g['attn.v_proj_weight'],
        ], dim=0)
    if 'attn.in_proj_bias' in torch_names and 'attn.bias_q_in' in g:
        g['attn.in_proj_bias'] = torch.cat([
            g['attn.bias_q_in'],
            g['attn.bias_k_in'],
            g['attn.bias_v_in'],
        ], dim=0)
    if 'attn.out_proj_weight' in g:
        g['attn.out_proj.weight'] = g['attn.out_proj_weight']
    if 'attn.out_proj_bias' in g:
        g['attn.out_proj.bias'] = g['attn.out_proj_bias']
    return g


def report(name, torch_grad, ai3_grad, mes, atol, rtol):
    if torch_grad is None or ai3_grad is None:
        print(f'  missing gradient for {name} on {mes} '
              f'(torch: {torch_grad is not None}, ai3: {ai3_grad is not None})')
        return False
    if torch.allclose(torch_grad, ai3_grad, atol=atol, rtol=rtol):
        print(f'  same gradients for {name} on {mes}')
        return True
    print(f'Gradients for {name} on {mes} differ')
    print('first 10 torch:', ' '.join(
        map(str, torch_grad.flatten()[:10].tolist())))
    print('first 10 ai3:', ' '.join(
        map(str, ai3_grad.flatten()[:10].tolist())))
    return False


def test_with(input, model, op, mes, *, dtype=torch.float32,
              atol=None, rtol=None):
    if atol is None:
        atol = 1e-6 if dtype == torch.float64 else 1e-3
    if rtol is None:
        rtol = 1e-6 if dtype == torch.float64 else 1e-3
    mes = f'{mes} ({dtype})'

    model = model.to(dtype)
    inputs = tuple(t.to(dtype) for t in as_tuple(input))
    for t in inputs:
        t.requires_grad = True

    out = model(*inputs)
    target = torch.randn(out.shape, dtype=dtype)

    grad_torch, input_grad_torch = get_grad(model, inputs, target)
    if op == CONV:
        ai3.swap_conv2d(model)
    elif op == MHA:
        ai3.swap_mha(model)
    grad_ai3, input_grad_ai3 = get_grad(model, inputs, target)
    if op == MHA:
        grad_ai3 = remap_mha_ai3(grad_ai3, set(grad_torch))

    same_gradients = True
    for name in grad_torch:
        same_gradients &= report(
            name, grad_torch[name], grad_ai3.get(name), mes, atol, rtol)

    if op == MHA:
        for label, tg, ag in zip(
                ('dq', 'dk', 'dv'), input_grad_torch, input_grad_ai3):
            same_gradients &= report(label, tg, ag, mes, atol, rtol)

    if same_gradients:
        print(f'Gradients are the same for {op} on {mes}')
    else:
        print(f'Gradients are different for {op} on {mes}')


def conv2d():
    class ConvModel(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super(ConvModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size)

        def forward(self, x):
            x = self.conv1(x)
            return x

    cases = [
        (lambda: torch.randn(10, 300, 300),
         lambda: ConvModel(10, 32, 3), 'no batch'),
        (lambda: torch.randn(1, 3, 224, 224),
         lambda: ConvModel(3, 16, (4, 3)), 'batch = 1'),
        (lambda: torch.randn(10, 10, 512, 52),
         lambda: ConvModel(10, 5, 5), 'batch = 10'),
    ]
    for dtype in (torch.float32, torch.float64):
        for make_input, make_model, mes in cases:
            test_with(make_input(), make_model(), CONV, mes, dtype=dtype)


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

    cases = [
        (lambda: (torch.randn(10, 50, 512),
                  torch.randn(10, 50, 512),
                  torch.randn(10, 50, 512)),
         lambda: MHAModel(512), 'basic'),
        (lambda: (torch.randn(50, 10, 512),
                  torch.randn(50, 10, 512),
                  torch.randn(50, 10, 512)),
         lambda: MHAModel(512, batch_first=False, bias=False),
         'basic no batch first no bias'),
        (lambda: (torch.randn(10, 50, 300),
                  torch.randn(10, 50, 200),
                  torch.randn(10, 50, 150)),
         lambda: MHAModel(300, 5, 200, 150), 'different kdim and vdim'),
        (lambda: (torch.randn(10, 60, 80),
                  torch.randn(10, 60, 80),
                  torch.randn(10, 60, 80)),
         lambda: MHAModel(80, 5, add_bias_kv=True, add_zero_attn=False),
         'with bias_kv'),
        (lambda: (torch.randn(10, 60, 80),
                  torch.randn(10, 60, 80),
                  torch.randn(10, 60, 80)),
         lambda: MHAModel(80, 5, add_bias_kv=True, add_zero_attn=True),
         'with bias_kv and add_zero_attn'),
    ]
    for dtype in (torch.float32, torch.float64):
        for make_input, make_model, mes in cases:
            test_with(make_input(), make_model(), MHA, mes, dtype=dtype)
