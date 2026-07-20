// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>

template <typename dtype>
Tensor mha::standard(Tensor query, Tensor key, Tensor value,
                     const mha::MemFormat input_format, const Tensor &q_proj,
                     const Tensor &k_proj, const Tensor &v_proj,
                     const std::optional<const Tensor> &qbias_in,
                     const std::optional<const Tensor> &k_bias_in,
                     const std::optional<const Tensor> &v_bias_in,
                     const std::optional<const Tensor> &kbias,
                     const std::optional<const Tensor> &vbias,
                     const Tensor &out_proj,
                     const std::optional<const Tensor> &out_bias,
                     const bool add_zero_attn, const uint num_heads,
                     const float dropout,
                     const std::optional<const Tensor> &key_padding_mask,
                     const std::optional<const Tensor> &attn_mask,
                     const bool need_weights, const bool average_attn_weights,
                     const bool is_causal, const bool need_to_project) {
    errs::bail("plain mha not implemented");
}

template <typename dtype>
std::array<std::optional<Tensor>, mha::NUM_GRAD> mha::standard_backward(
    const intptr_t dout_address, Tensor query, Tensor key, Tensor value,
    const mha::MemFormat input_format, const Tensor &q_proj,
    const Tensor &k_proj, const Tensor &v_proj,
    const std::optional<const Tensor> &q_bias_in,
    const std::optional<const Tensor> &k_bias_in,
    const std::optional<const Tensor> &v_bias_in,
    const std::optional<const Tensor> &k_bias,
    const std::optional<const Tensor> &v_bias, const Tensor &out_proj,
    const std::optional<const Tensor> &out_bias, const bool add_zero_attn,
    const uint num_heads, const float dropout,
    const std::optional<const Tensor> &key_padding_mask,
    const std::optional<const Tensor> &attn_mask, const bool need_weights,
    const bool average_attn_weights, const bool is_causal,
    const bool need_to_project_input) {
    errs::bail("plain mha backward not implemented");
}

template Tensor mha::standard<float>(MHA_PARAMS);
template Tensor mha::standard<double>(MHA_PARAMS);
template std::array<std::optional<Tensor>, mha::NUM_GRAD>
mha::standard_backward<float>(const intptr_t, MHA_PARAMS);
template std::array<std::optional<Tensor>, mha::NUM_GRAD>
mha::standard_backward<double>(const intptr_t, MHA_PARAMS);
