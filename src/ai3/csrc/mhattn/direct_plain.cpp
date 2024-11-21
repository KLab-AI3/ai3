// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>

template <typename dtype>
Tensor mha::direct(Tensor query, Tensor key, Tensor value, Tensor q_proj,
                   Tensor k_proj, Tensor v_proj, Tensor qbias_in,
                   Tensor k_bias_in, Tensor v_bias_in, Tensor kbias,
                   Tensor vbias, Tensor out_proj, Tensor out_bias,
                   const uint num_heads, const uint head_dim,
                   std::optional<Tensor> key_padding_mask,
                   std::optional<Tensor> attn_mask, const bool need_weights,
                   const bool average_attn_weights, const bool is_causal) {}
