import torch
import torchvision
import ai3


def conv2d_selector(orig, input_shape) -> str:
    out_channels = orig.weight.shape[0]
    if (out_channels < 50 and
        input_shape[1] < 50 and
        input_shape[1] > 150 and
            input_shape[2] > 150):
        return 'direct'
    return 'smm'


input_shape = (1, 3, 224, 224)
input_data = torch.randn(input_shape)
vgg16 = torchvision.models.vgg16(
    weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16.eval()
with torch.inference_mode():
    orig_out = vgg16(input_data)
    model: ai3.Model = ai3.convert(
        vgg16, {'conv2d': conv2d_selector,
                'maxpool2d': 'default'},
        input_shape)
    sb_out = model(input_data)
    ai3.swap_operation(torch.nn.Conv2d, vgg16, [
                       'direct', 'smm'] * 8, input_shape)
    sc_out = vgg16(input_data)
    assert torch.allclose(
        orig_out, sb_out, atol=1e-4)
    assert torch.allclose(
        orig_out, sc_out, atol=1e-4)
