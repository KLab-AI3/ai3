import torch
from torch import nn
import ai3


class AttentionNet(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2):
        super(AttentionNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim, num_heads, bias=True, batch_first=True)
            for _ in range(num_layers)])

    def forward(self, x):
        for attn in self.layers:
            x = attn(x, x, x, need_weights=False)[0]
        return x


if __name__ == '__main__':
    input_data = torch.randn(4, 16, 64)
    orig = AttentionNet()
    orig.eval()
    with torch.inference_mode():
        torch_out = orig(input_data)
        model: ai3.Model = ai3.convert(orig)
        sb_out = model(input_data)
        ai3.swap_mha(orig)
        sc_out = orig(input_data)
    assert torch.allclose(torch_out, sb_out, atol=1e-3)
    assert torch.allclose(torch_out, sc_out, atol=1e-3)
