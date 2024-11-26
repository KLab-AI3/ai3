// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>
#include <cudnn_frontend.h>

template <typename dtype>
Tensor mhattn::standard(
    Tensor query, Tensor key, Tensor value, const Tensor &q_proj,
    const Tensor &k_proj, const Tensor &v_proj,
    const std::optional<const Tensor> &qbias_in,
    const std::optional<const Tensor> &k_bias_in,
    const std::optional<const Tensor> &v_bias_in,
    const std::optional<const Tensor> &kbias,
    const std::optional<const Tensor> &vbias, const Tensor &out_proj,
    const std::optional<const Tensor> &out_bias, const uint num_heads,
    const uint head_dim, std::optional<Tensor> &key_padding_mask,
    std::optional<Tensor> &attn_mask, const bool need_weights,
    const bool average_attn_weights, const bool is_causal) {
    ensure_same_type(query, key, value);

    cudnnHandle_t handle = (cudnnHandle_t)Context::cudnn_handle_t();
}

template Tensor mhattn::standard<float>(MHATTN_PARAMS);
template Tensor mhattn::standard<double>(MHATTN_PARAMS);
