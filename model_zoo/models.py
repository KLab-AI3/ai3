import torchvision.models as tvm
from example.manual_conv2d import ConvNet
from example.manual_mha import AttentionNet


def alexnet():
    return tvm.alexnet(), (3, 224, 224), 'alexnet'


def convnext():
    return tvm.convnext_base(weights='DEFAULT'), (3, 224, 224), 'convnext'


def densenet():
    return tvm.DenseNet(), (3, 224, 224), 'densenet'


def efficientnet():
    return tvm.efficientnet_v2_s(
        weights=tvm.EfficientNet_V2_S_Weights.DEFAULT), (
        3, 224, 224), 'efficientnet'


def googlenet():
    return tvm.googlenet(
        weights=tvm.GoogLeNet_Weights.DEFAULT), (
        3, 224, 224), 'googlenet'


def inception():
    return tvm.inception_v3(
        weights=tvm.Inception_V3_Weights.DEFAULT), (
        3, 224, 224), 'inception'


def maxvit():
    return tvm.maxvit_t(), (3, 224, 224), 'maxvit'


def mnasnet():
    return tvm.mnasnet1_0(
        weights=tvm.MNASNet1_0_Weights.DEFAULT), (
        3, 224, 224), 'mnasnet'


def mobilenet():
    return tvm.mobilenet_v3_large(), (3, 224, 224), 'mobilenet'


def regnet():
    return tvm.regnet_y_16gf(), (3, 224, 224), 'regnet'


def resnet():
    return tvm.resnet152(), (3, 224, 224), 'resnet'


def shufflenet():
    return tvm.shufflenet_v2_x2_0(), (3, 224, 224), 'shufflenet'


def manual_conv2d():
    return ConvNet(), (3, 224, 224), 'manual conv2d'


def squeezenet():
    return tvm.squeezenet1_1(), (3, 224, 224), 'squeezenet'


def swintransformer():
    return tvm.swin_b(), (3, 224, 224), 'swin transformer'


def vgg16():
    return tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT), (3, 224, 224), 'vgg16'


def visiontransformer():
    return tvm.vit_b_16(), (3, 224, 224), 'vision transformer'


def manual_mha():
    return AttentionNet(), (16, 64), 'manual mha'
