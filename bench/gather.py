from collections import defaultdict
import sys
from torch import nn
import time
import torch
import matplotlib.pyplot as plt
import ai3
import os
import torchvision.models as tvm
import pickle
import numpy

N = 10
WARM_ITERS = 10
AVG_OVER = 10
EARLY_SHAPE = (N, 3, 224, 224)
EARLY_MIDDLE_SHAPE = (N, 64, 112, 112)
LATE_MIDDLE_SHAPE = (N, 256, 28, 28)
LATE_SHAPE = (N, 512, 14, 14)

CUDNN_SUFFIX = 'cudnn'
EARLY_FNAME = 'early'
EARLY_MIDDLE_FNAME = 'early middle'
LATE_MIDDLE_FNAME = 'data_late_middle'
LATE_FNAME = 'data_late'


RESULT_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'results')
FILE_SUFFIX = CUDNN_SUFFIX
SAVE_TO_DIR = os.path.join(RESULT_DIR, CUDNN_SUFFIX)
TORCH_NAME = 'torch cuDNN'
ALGOS_TO_USE_CONV2D = [
    ('implicit gemm', 'implicit GEMM cuDNN'),
    ('implicit precomp gemm', 'implicit precomp GEMM cuDNN'),
    ('gemm', 'GEMM cuDNN'),
    ('winograd', 'Winograd cuDNN'),
    ('direct', 'direct SYCL'),
    ('smm', 'SMM SYCL')]
ALGOS_TO_USE_MODEL = [
    ('implicit gemm', 'implicit GEMM cuDNN'),
    ('implicit precomp gemm', 'implicit precomp GEMM cuDNN'),
    ('gemm', 'GEMM cuDNN'),
    ('direct', 'direct SYCL'),
    ('smm', 'SMM SYCL')]

os.makedirs(SAVE_TO_DIR, exist_ok=True)
plt.rcParams['savefig.dpi'] = 500


def warmup(runner, data):
    for _ in range(WARM_ITERS):
        runner(data)


def time_forward(runner, data):
    start = time.time()
    runner(data)
    end = time.time()
    return end - start


class Conv2D(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


def conv2d_times(input):
    times_for_layer = defaultdict(float)
    orig = Conv2D(
        input.shape[1], input.shape[1], 3)
    orig.eval()
    warmup(orig, input)
    times_for_layer[TORCH_NAME] = sum(
        [time_forward(orig, input) for _ in range(0, AVG_OVER)]) / AVG_OVER

    for (algo, name) in ALGOS_TO_USE_CONV2D:
        swap = ai3.convert(orig, {'conv2d': algo})
        warmup(swap, input)
        times_for_layer[name] = sum(
            [time_forward(swap, input) for _ in range(0, AVG_OVER)]) / AVG_OVER

    return times_for_layer


def save_conv2d(data_early, data_early_middle, data_late_middle,
                data_late):
    data = {EARLY_FNAME: data_early,
            EARLY_MIDDLE_FNAME: data_early_middle,
            LATE_MIDDLE_FNAME: data_late_middle,
            LATE_FNAME: data_late}
    for file, data in data.items():
        with open(os.path.join(SAVE_TO_DIR, f'{file}_{FILE_SUFFIX}.pickle'), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_conv2d_data(general_fname):
    with open(os.path.join(RESULT_DIR, CUDNN_SUFFIX, f'{general_fname}_{CUDNN_SUFFIX}.pickle'), 'rb') as handle:
        data = pickle.load(handle)

    return data


def conv2d_save_graph():
    data_early = get_conv2d_data(EARLY_FNAME)
    data_early_middle = get_conv2d_data(EARLY_MIDDLE_FNAME)
    data_late_middle = get_conv2d_data(LATE_MIDDLE_FNAME)
    data_late = get_conv2d_data(LATE_FNAME)
    plt.figure(figsize=(12, 6))
    colors = ['lightblue', 'peachpuff', 'lightgreen', 'lightcoral', 'thistle',
              'burlywood', 'lightpink', 'palegoldenrod', 'paleturquoise',
              'lightgray']
    algos = list(data_early.keys())
    input_shape_labels = [EARLY_SHAPE, EARLY_MIDDLE_SHAPE,
                          LATE_MIDDLE_SHAPE, LATE_SHAPE]

    bar_width = 0.1
    x = range(len(input_shape_labels))

    for i, algo in enumerate(algos):
        plt.bar([pos + i * bar_width for pos in x],
                [data_early[algo], data_early_middle[algo],
                    data_late_middle[algo], data_late[algo]],
                width=bar_width,
                color=colors[i % len(colors)],
                label=algo)

    plt.xlabel(
        'Input Shapes (N, C, H, W)', fontsize=14)
    plt.ylabel('Time (s)', fontsize=16)
    plt.title(
        f'Latency of Conv2D Operation', fontsize=18)
    plt.yticks(numpy.linspace(0, 0.03, 11), fontsize=13)
    plt.xticks([pos + (len(algos) - 1) * bar_width /
               2 for pos in x], input_shape_labels, fontsize=13)
    plt.legend(fontsize=13, loc='upper center',
               bbox_to_anchor=(0.5, -0.1), ncol=4)

    plt.savefig(
        os.path.join(RESULT_DIR, f'conv2d_times.png'),
        bbox_inches='tight')
    plt.close()


def model_times(model, input):
    times_for_model = defaultdict(float)
    model.eval()

    warmup(model, input)
    times_for_model[TORCH_NAME] = sum(
        [time_forward(model, input) for _ in range(0, AVG_OVER)]) / AVG_OVER

    for (algo, name) in ALGOS_TO_USE_MODEL:
        ai3.swap_conv2d(model, algo)
        warmup(model, input)
        times_for_model[name] = sum(
            [time_forward(model, input) for _ in range(0, AVG_OVER)]) / AVG_OVER

    return times_for_model


def save_model_times(models_data):
    normalized = {}
    for model, algo_times in models_data.items():
        normed_model = {
            algo: time / algo_times[TORCH_NAME] for algo,
            time in algo_times.items() if algo != TORCH_NAME}
        normalized[model] = normed_model

    with open(os.path.join(SAVE_TO_DIR, f'models_{FILE_SUFFIX}.pickle'), 'wb') as handle:
        pickle.dump(normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)


def table_save():
    with open(os.path.join(RESULT_DIR, CUDNN_SUFFIX, f'models_{CUDNN_SUFFIX}.pickle'), 'rb') as handle:
        cudnn_models = pickle.load(handle)

    models_data = {}
    for model in set(cudnn_models.keys()).union(cudnn_models.keys()):
        models_data[model] = {}
        models_data[model].update(cudnn_models[model])

    columns = list(models_data[next(iter(models_data))].keys())
    row_labels = list(models_data.keys())
    table_data = []

    for model in row_labels:
        row = [round(models_data[model].get(col, 0), 4) for col in columns]
        table_data.append(row)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    table = ax.table(cellText=table_data, colLabels=columns,
                     rowLabels=row_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.auto_set_column_width(col=list(range(len(columns))))
    table.scale(1, 2)

    ax.annotate(
        'Relative to PyTorch Using cuDNN', xy=(0, 0.05),
        fontsize=16,
        bbox=dict(
            boxstyle='round,pad=0.3', edgecolor='black', facecolor='none'))

    plt.savefig(os.path.join(RESULT_DIR, 'model_times.png'),
                bbox_inches='tight')
    plt.close()


def figures():
    conv2d_save_graph()
    table_save()


def gather():
    with torch.inference_mode():
        input = torch.randn(EARLY_SHAPE)
        print('conv2d early')
        conv2d_times_early = conv2d_times(input)
        print(conv2d_times_early)

        input = torch.randn(EARLY_MIDDLE_SHAPE)
        print('conv2d early middle')
        conv2d_times_early_middle = conv2d_times(
            input)
        print(conv2d_times_early_middle)

        input = torch.randn(LATE_MIDDLE_SHAPE)
        print('conv2d late middle')
        conv2d_times_late_middle = conv2d_times(
            input)
        print(conv2d_times_late_middle)

        input = torch.randn(LATE_SHAPE)
        print('conv2d channels')
        conv2d_times_late = conv2d_times(input)
        print(conv2d_times_late)

        save_conv2d(conv2d_times_early, conv2d_times_early_middle,
                    conv2d_times_late_middle, conv2d_times_late)

        input = torch.randn(EARLY_SHAPE)
        orig_models = {'AlexNet': tvm.alexnet(),
                       'DenseNet': tvm.DenseNet(),
                       'GoogLeNet': tvm.googlenet(),
                       'Inception V3': tvm.inception_v3(),
                       'ResNet152': tvm.resnet152(),
                       'SqueezeNet 1.1': tvm.squeezenet1_1(),
                       'Swin Transformer Base': tvm.swin_b(),
                       'VGG16': tvm.vgg16(),
                       'Vision Transformer Base 16': tvm.vit_b_16()}
        models_data = {}

        for model_name, model in orig_models.items():
            print(model_name)
            models_data[model_name] = model_times(
                model, input)
            print(f'  {models_data[model_name]}')

        save_model_times(models_data)


if __name__ == '__main__':
    make_figures = False
    if len(sys.argv) > 1 and sys.argv[1] == 'figures':
        make_figures = True

    if make_figures:
        figures()
    else:
        assert ai3.using_cudnn()
        gather()
