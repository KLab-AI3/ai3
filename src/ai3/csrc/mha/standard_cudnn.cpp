#include <ai3.hpp>
#include <algos.hpp>
#include <cudnn.h>
#include <cudnn_utils.hpp>

template <typename dtype>
Tensor
mha::standard(Tensor query, Tensor key, Tensor value, const Tensor &q_proj,
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
    errs::bail("mha not implemented outside of cuDNN");
}

template Tensor mha::standard<float>(MHA_PARAMS);
template Tensor mha::standard<double>(MHA_PARAMS);
