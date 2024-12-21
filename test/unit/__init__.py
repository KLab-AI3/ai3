import math


def pooling_poss_output_size(
    input_height, input_width, padding, stride, kernel_height,
        kernel_width, ceil_mode):
    top_height = input_height + 2 * \
        padding[0] - kernel_height
    bottom_height = stride[0]

    if ceil_mode:
        poss_output_height = math.ceil(
            top_height / bottom_height) + 1
    else:
        poss_output_height = top_height // bottom_height + 1

    top_width = input_width + 2 * \
        padding[1] - kernel_width
    bottom_width = stride[1]

    if ceil_mode:
        poss_output_width = math.ceil(
            top_width / bottom_width) + 1
    else:
        poss_output_width = top_width // bottom_width + 1

    return poss_output_height, poss_output_width


def run():
    print('UNIT')
    from . import conv2d, maxpool2d, linear, relu, avgpool2d, adaptiveavgpool2d, flatten, custom, mha
    [m.main() for m in [conv2d, maxpool2d, linear, relu, avgpool2d, adaptiveavgpool2d, flatten, custom, mha]]
