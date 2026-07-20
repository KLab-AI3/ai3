#pragma once

#include "algos.hpp"
#include <ai3.hpp>
#include <optional>

/**
 * @DEFAULT_BOOL{ReLU}
 */
const bool CUSTOM_DEFAULT_MHA = false;
/**
 * @ingroup custom
 * Whether the custom implementation handles inputs (projects, concats bias,
 * adds zero attention)
 */
const std::optional<bool> CUSTOM_MHA_HANDLES_INPUTS = std::nullopt;
/**
 * @ingroup custom
 * Whether the custom implementation projects outputs
 */
const std::optional<bool> CUSTOM_MHA_PROJECTS_OUTPUT = std::nullopt;

/**
 * @CUSTOM_OP{MultiHead Attention,mha}
 */
template <typename dtype>
Tensor mha_custom(
    Tensor query, Tensor key, Tensor value, const mha::MemFormat input_format,
    const Tensor &q_proj, const Tensor &k_proj, const Tensor &v_proj,
    const std::optional<const Tensor> &q_bias_in,
    const std::optional<const Tensor> &k_bias_in,
    const std::optional<const Tensor> &v_bias_in,
    const std::optional<const Tensor> &k_bias,
    const std::optional<const Tensor> &v_bias, const Tensor &out_proj,
    const std::optional<const Tensor> &out_bias, const bool add_zero_attn,
    const uint num_heads, const float dropout,
    const std::optional<const Tensor> &key_padding_mask,
    const std::optional<const Tensor> &attn_mask, const bool need_weights,
    const bool average_attn_weights, const bool is_causal) {
    errs::bail_if(!CUSTOM_MHA_HANDLES_INPUTS.has_value(),
                  "unknown if custom implementation supports QKV projection");
    errs::bail_if(
        !CUSTOM_MHA_PROJECTS_OUTPUT.has_value(),
        "unknown if custom implementation supports projecting output");
    errs::no_user_def("multihead attention");
}

/**
 * @CUSTOM_OP{MultiHead Attention Backward,mha}
 */
template <typename dtype>
std::array<std::optional<Tensor>, mha::NUM_GRAD> mha_custom_backward(
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
    const bool average_attn_weights, const bool is_causal) {
    errs::no_user_def("multihead attention backward");
}
