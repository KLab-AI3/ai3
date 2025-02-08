// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ai3.hpp>
#include <cuda_utils.hpp>
#include <cudnn_utils.hpp>

template <typename dtype>
Tensor conv_bias_forward_with_algo(Tensor input, const Tensor &kernel,
                                   const std::optional<const Tensor> &bias,
                                   const uint padding_h, const uint padding_w,
                                   const uint stride_h, const uint stride_w,
                                   const uint dilation_h, const uint dilation_w,
                                   const PaddingMode padding_mode, uint groups,
                                   cudnnConvolutionFwdAlgo_t algo,
                                   const bool guess = false) {
    ensure_same_type(input, kernel, bias);
    errs::bail_if(padding_mode != PaddingMode::Zeros,
                  "padding mode must be zeroes");
    errs::bail_if(groups != 1, "groups must be 1");

    cudnnHandle_t handle = (cudnnHandle_t)Context::cudnn_handle_t();
    cudaStream_t cpy_stream;
    CUDA_CHECK(cudaStreamCreate(&cpy_stream));

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    const uint input_channels = input.input_channels();
    const uint input_height = input.height();
    const uint input_width = input.width();
    const uint kernel_height = kernel.height();
    const uint kernel_width = kernel.width();
    const uint output_channels = kernel.output_channels();

    const uint output_height = output_hw_for_2d<dtype>(
        input_height, kernel_height, padding_h, dilation_h, stride_h, false);
    const uint output_width = output_hw_for_2d<dtype>(
        input_width, kernel_width, padding_w, dilation_w, stride_w, false);

    uint num_samples;
    Tensor output;
    if (input.batched(sample_dims::CONV2D)) {
        num_samples = input.batch_size(sample_dims::CONV2D);
        output =
            Tensor({num_samples, output_channels, output_height, output_width},
                   input.scalar_type);
    } else {
        num_samples = 1;
        output = Tensor({output_channels, output_height, output_width},
                        input.scalar_type);
    }

    cudnnDataType_t cudnn_dtype = cudnn_data_type<dtype>();

    dtype *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, input.count() * sizeof(dtype)));
    CUDA_CHECK(cudaMemcpyAsync(d_input, input.data,
                               input.count() * sizeof(dtype),
                               cudaMemcpyHostToDevice, cpy_stream));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, num_samples, input_channels,
        input_height, input_width));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_desc));
    dtype *d_kernel;
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        kernel_desc, cudnn_dtype, CUDNN_TENSOR_NCHW, output_channels,
        input_channels, kernel_height, kernel_width));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel.count() * sizeof(dtype)));
    CUDA_CHECK(cudaMemcpyAsync(d_kernel, kernel.data,
                               kernel.count() * sizeof(dtype),
                               cudaMemcpyHostToDevice, cpy_stream));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    dtype *d_output;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, num_samples,
        output_channels, output_height, output_width));
    CUDA_CHECK(cudaMalloc(&d_output, output.count() * sizeof(dtype)));

    dtype *d_bias = nullptr;
    const bool with_bias = bias.has_value();
    cudnnTensorDescriptor_t bias_desc;
    if (with_bias) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                               cudnn_dtype, 1, output_channels,
                                               1, 1));
        CUDA_CHECK(cudaMalloc(&d_bias, bias->count() * sizeof(dtype)));
        CUDA_CHECK(cudaMemcpyAsync(d_bias, bias->data,
                                   bias->count() * sizeof(dtype),
                                   cudaMemcpyHostToDevice, cpy_stream));
    }

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, dilation_h,
        dilation_w, CUDNN_CROSS_CORRELATION, cudnn_dtype));

    if (guess) {
        int returnedAlgoCount;
        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        cudnnConvolutionFwdAlgoPerf_t perfResults[requestedAlgoCount];
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle, input_desc, kernel_desc, conv_desc, output_desc,
            requestedAlgoCount, &returnedAlgoCount, perfResults));
        algo = perfResults[0].algo;
    }

    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, kernel_desc, conv_desc, output_desc, algo,
        &workspace_bytes));
    void *d_workspace;

    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));

    const dtype alpha = 1.0, beta = 0.0;

    CUDA_CHECK(cudaStreamSynchronize(cpy_stream));
    CUDNN_CHECK(cudnnConvolutionForward(
        handle, &alpha, input_desc, d_input, kernel_desc, d_kernel, conv_desc,
        algo, d_workspace, workspace_bytes, &beta, output_desc, d_output));
    if (with_bias) {
        CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bias_desc, d_bias, &alpha,
                                   output_desc, d_output));
    }

    CUDA_CHECK(cudaMemcpyAsync(output.data, d_output,
                               output.count() * sizeof(dtype),
                               cudaMemcpyDeviceToHost, cpy_stream));

    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    if (with_bias) {
        CUDA_CHECK(cudaFree(d_bias));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(kernel_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    CUDA_CHECK(cudaStreamSynchronize(cpy_stream));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(cpy_stream));

    return output;
}
