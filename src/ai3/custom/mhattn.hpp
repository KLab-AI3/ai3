#pragma once

#include <ai3.hpp>

/**
 * @DEFAULT_BOOL{ReLU}
 */
const bool DEFAULT_MHATTN = true;

/**
 * @CUSTOM_OP{ReLU,relu}
 */
template <typename dtype>
Tensor
mhattn_custom(Tensor query, Tensor key, Tensor value, const Tensor &q_proj,
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
    errs::no_user_def("multihead attention");
}
