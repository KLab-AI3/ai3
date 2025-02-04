// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>
#include <cublas_utils.hpp>
#include <cuda_utils.hpp>
#include <optional>

template <typename dtype>
Tensor linear::gemm(Tensor input, const Tensor &weight,
                    const std::optional<const Tensor> &bias) {
    ensure_same_type(input, weight);
    errs::bail_if(input.width() != weight.width(),
                  "Invalid matrix multiplication: input width=", input.width(),
                  " weight width=", weight.width());

    const uint in_features = input.width();
    const uint out_features = weight.height();
    const bool has_bias = bias.has_value();

    Tensor output;
    uint num_samples;
    if (input.batched(sample_dims::LINEAR)) {
        num_samples = input.batch_size(sample_dims::LINEAR);
        output = Tensor({num_samples, out_features}, input.scalar_type);
    } else {
        num_samples = 1;
        output = Tensor({out_features}, input.scalar_type);
    }

    dtype *d_input, *d_weight, *d_bias, *d_output;
    uint input_size = input.count() * sizeof(dtype);
    uint weight_size = weight.count() * sizeof(dtype);
    uint output_size = output.count() * sizeof(dtype);
    uint bias_size = has_bias ? bias->count() * sizeof(dtype) : 0;

    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size));
    if (has_bias) {
        CUDA_CHECK(cudaMalloc(&d_bias, bias_size));
    }
    CUDA_CHECK(cudaMalloc(&d_output, output_size));

    cudaStream_t cpy_stream;
    CUDA_CHECK(cudaStreamCreate(&cpy_stream));

    CUDA_CHECK(cudaMemcpyAsync(d_input, input.data, input_size,
                               cudaMemcpyHostToDevice, cpy_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weight, weight.data, weight_size,
                               cudaMemcpyHostToDevice, cpy_stream));
    if (has_bias) {
        CUDA_CHECK(cudaMemcpyAsync(d_bias, bias->data, bias_size,
                                   cudaMemcpyHostToDevice, cpy_stream));
    }

    cublasHandle_t handle = (cublasHandle_t)Context::cublas_handle_t();

    const dtype alpha = 1.0, beta = 0.0;
    cudaDataType cublas_dtype = cuda_data_type<dtype>();

    CUDA_CHECK(cudaStreamSynchronize(cpy_stream));
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features,
                              num_samples, in_features, &alpha, d_weight,
                              cublas_dtype, in_features, d_input, cublas_dtype,
                              in_features, &beta, d_output, cublas_dtype,
                              out_features, cublas_dtype, CUBLAS_GEMM_DEFAULT));

    if (has_bias) {
        for (int i = 0; i < num_samples; ++i) {
            CUBLAS_CHECK(cublasAxpyEx(handle, out_features, &alpha,
                                      cublas_dtype, d_bias, cublas_dtype, 1,
                                      d_output + i * out_features, cublas_dtype,
                                      1, cublas_dtype));
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(output.data, d_output, output_size,
                               cudaMemcpyDeviceToHost, cpy_stream));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    if (bias.has_value()) {
        CUDA_CHECK(cudaFree(d_bias));
    }

    CUDA_CHECK(cudaStreamSynchronize(cpy_stream));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(cpy_stream));

    return output;
}

template Tensor linear::gemm<float>(LINEAR_PARAMS);
template Tensor linear::gemm<double>(LINEAR_PARAMS);
