#pragma once

#include "ai3.hpp"
#include <optional>

template <typename dtype>
Tensor<dtype> _maxpool2d(Tensor<dtype> input, const uint kernel_h,
                         const uint kernel_w, const uint padding_h,
                         const uint padding_w, const uint stride_h,
                         const uint stride_w, const uint dilation_h,
                         const uint dilation_w, const bool ceil_mode) {
    const uint input_channels = input.input_channels();
    const uint input_height = input.height();
    const uint input_width = input.width();

    const uint output_channels = input_channels;

    const uint output_height = output_hw_for_2d<dtype>(
        input_height, kernel_h, padding_h, dilation_h, stride_h, ceil_mode);
    const uint output_width = output_hw_for_2d<dtype>(
        input_width, kernel_w, padding_w, dilation_w, stride_w, ceil_mode);

    Tensor<dtype> output;
    uint num_samples;
    if (input.batched(input_dims::POOL2D)) {
        num_samples = input.batch_size(input_dims::POOL2D);
        output = Tensor<dtype>(
            {num_samples, output_channels, output_height, output_width});
    } else {
        num_samples = 1;
        output = Tensor<dtype>({output_channels, output_height, output_width});
    }

    const dtype lowest = std::numeric_limits<dtype>::lowest();
    for (uint samp = 0; samp < num_samples; samp++) {
        for (uint out_c = 0; out_c < output_channels; out_c++) {
            for (uint out_h = 0; out_h < output_height; out_h++) {
                for (uint out_w = 0; out_w < output_width; out_w++) {
                    dtype cur_max = lowest;
                    for (uint kern_r = 0; kern_r < kernel_h; ++kern_r) {
                        for (uint kern_c = 0; kern_c < kernel_w; ++kern_c) {
                            uint h_offset = out_h * stride_h - padding_h +
                                            kern_r * dilation_h;
                            uint w_offset = out_w * stride_w - padding_w +
                                            kern_c * dilation_w;

                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                dtype cur = input.data[to_linear(
                                    samp, out_c, h_offset, w_offset,
                                    output_channels, input_height,
                                    input_width)];
                                if (cur > cur_max) {
                                    cur_max = cur;
                                }
                            }
                        }
                    }
                    output.data[to_linear(samp, out_c, out_h, out_w,
                                          output_channels, output_height,
                                          output_width)] = cur_max;
                }
            }
        }
    }

    return output;
}
