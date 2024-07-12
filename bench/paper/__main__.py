from collections import defaultdict
import time
import torch
from bench import warm_up
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import ai3
import os
import torchvision.models as tvm

CUDA_AVAILABLE = torch.cuda.is_available()
result_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "results")
if CUDA_AVAILABLE:
    SAVE_TO_DIR = os.path.join(result_dir, "gpu")
else:
    SAVE_TO_DIR = os.path.join(result_dir, "cpu")
os.makedirs(SAVE_TO_DIR, exist_ok=True)
plt.rcParams['savefig.dpi'] = 500


def time_forward(runner, data, should_warm_up=False, warm_iters=1):
    if should_warm_up:
        for _ in range(warm_iters):
            warm_up(runner, data)
    start = time.time()
    runner(data)
    end = time.time()
    return end - start


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


def gather_conv2d_times(input):
    times_for_layer = defaultdict(float)
    orig = Conv2D(input.shape[1], input.shape[1], 3)
    orig.eval()
    torch_name = "torch"
    times_for_layer[torch_name] = time_forward(orig, input, True)

    swap_ip_gemm = ai3.swap_backend(orig, {"conv2d": "implicit precomp gemm"})
    times_for_layer["implicit precomp GEMM"] = time_forward(
        swap_ip_gemm, input, True)
    swap_i_gemm = ai3.swap_backend(orig, {"conv2d": "implicit gemm"})
    times_for_layer["implicit GEMM"] = time_forward(
        swap_i_gemm, input, True)

    swap_wino = ai3.swap_backend(orig, {"conv2d": "winograd"})
    times_for_layer["winograd"] = time_forward(
        swap_wino, input, True)

    swap_gemm = ai3.swap_backend(orig, {"conv2d": "gemm"})
    times_for_layer["GEMM"] = time_forward(
        swap_gemm, input, True)

    swap_guess = ai3.swap_backend(orig, {"conv2d": "guess"})
    times_for_layer["guess"] = time_forward(
        swap_guess, input, True)

    return times_for_layer


N = 10
EARLY_SHAPE = (N, 3, 224, 224)
EARLY_MIDDLE_SHAPE = (N, 64, 112, 112)
LATE_MIDDLE_SHAPE = (N, 256, 28, 28)
LATE_SHAPE = (N, 512, 14, 14)


def save_combined_plot(data_early, data_early_middle, data_late_middle, data_late):
    plt.figure(figsize=(12, 6))

    colors = ['lightblue', 'peachpuff', 'lightgreen', 'lightcoral', 'thistle',
              'burlywood', 'lightpink', 'lightgray', 'palegoldenrod',
              'paleturquoise']
    backends = list(data_early.keys())
    input_shape_labels = [f'{EARLY_SHAPE}', f'{EARLY_MIDDLE_SHAPE}', f'{LATE_MIDDLE_SHAPE}',
                          f'{LATE_SHAPE}']

    bar_width = 0.1
    x = range(len(input_shape_labels))

    for i, backend in enumerate(backends):
        plt.bar([pos + i * bar_width for pos in x],
                [data_early[backend], data_early_middle[backend], data_late_middle[backend], data_late[backend]],
                width=bar_width,
                color=colors[i % len(colors)],
                label=backend)

    plt.xlabel('Input Shapes (N, C, H, W)', fontsize=14)
    plt.ylabel('Time (s)', fontsize=14)
    plt.title('Latency of Conv2D Operation', fontsize=16)
    plt.xticks([pos + (len(backends) - 1) * bar_width / 2 for pos in x], input_shape_labels)
    plt.legend()

    plt.savefig(os.path.join(SAVE_TO_DIR, "combined_conv2d_times.png"), bbox_inches='tight')
    plt.close()

def gather_model_times(model, input):
    times_for_model = defaultdict(float)
    model.eval()
    torch_name = "torch"
    times_for_model[torch_name] = time_forward(model, input)

    # if CUDA_AVAILABLE:
    #     torch_comp = torch.compile(model)
    #     times_for_model["torch graph mode"] = time_forward(
    #         torch_comp, input, true, warm_iters=10)

    ai3.swap_conv2d(model, "implicit gemm")
    times_for_model["implicit gemm"] = time_forward(
        model, input)

    ai3.swap_conv2d(model, "implicit precomp gemm")
    times_for_model["implicit precomp GEMM"] = time_forward(
        model, input)

    ai3.swap_conv2d(model,  "guess")
    times_for_model["guess"] = time_forward(
        model, input)

    return times_for_model

def save_model_data_table(models_data):
    df = pd.DataFrame(models_data).transpose()
    df = df.round(4)

    norm_column = 'torch'
    df = df.div(df[norm_column], axis=0)
    df = df.drop(columns=[norm_column])

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                     cellLoc='center', loc='center')

    plt.savefig(os.path.join(SAVE_TO_DIR, 'model_times_table.png'), bbox_inches='tight')
    plt.close()


# ['SIMPLE_CREATED', 'VGG16', 'ALEXNET', 'DENSENET', 'GOOGLENET', 'INCEPTION', 'SQUEEZENET', 'VISION_TRANSFORMER', 'SWIN_TRANSFORMER', 'RESNET']
with torch.inference_mode():
    input = torch.randn(EARLY_SHAPE)
    print('conv2d early')
    conv2d_times_early = gather_conv2d_times(input)
    print(conv2d_times_early)

    input = torch.randn(EARLY_MIDDLE_SHAPE)
    print('conv2d early middle')
    conv2d_times_early_middle = gather_conv2d_times(input)
    print(conv2d_times_early_middle)

    input = torch.randn(LATE_MIDDLE_SHAPE)
    print('conv2d late middle')
    conv2d_times_late_middle = gather_conv2d_times(input)
    print(conv2d_times_late_middle)

    input = torch.randn(LATE_SHAPE)
    print('conv2d channels')
    conv2d_times_late = gather_conv2d_times(input)
    print(conv2d_times_late)

    save_combined_plot(conv2d_times_early, conv2d_times_early_middle, conv2d_times_late_middle, conv2d_times_late)

    input = torch.randn(EARLY_SHAPE)
    orig_models = {"AlexNet" :tvm.alexnet(),
                   "DenseNet" : tvm.DenseNet(),
                   "GoogleNet" : tvm.googlenet(),
                   "Incetion V3" : tvm.inception_v3(),
                   "ResNet152" : tvm.resnet152(),
                   "Squeezenet 1.1": tvm.squeezenet1_1(),
                   "Swin Transformer Base": tvm.swin_b(),
                   "VGG16" : tvm.vgg16(),
                   "Vision Transformer Base 16" : tvm.vit_b_16()}
    models_data = {}

    for model_name, model in orig_models.items():
        print(model_name)
        models_data[model_name] = gather_model_times(model, input)
        print(f"  {models_data[model_name]}")
    save_model_data_table(models_data)
