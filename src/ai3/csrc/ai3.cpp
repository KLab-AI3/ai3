// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../custom/adaptiveavgpool2d.hpp"
#include "../custom/avgpool2d.hpp"
#include "../custom/conv2d.hpp"
#include "../custom/flatten.hpp"
#include "../custom/linear.hpp"
#include "../custom/maxpool2d.hpp"
#include "../custom/mha.hpp"
#include "../custom/relu.hpp"

const std::string DEFAULT_OPT_STR = "default";
const std::string CUSTOM_OPT_STR = "custom";

class Layer {
  public:
    virtual Tensor _forward_float(Tensor input) {
        (void)input;
        errs::forward_not_implemented<Layer>();
    }
    virtual Tensor _forward_double(Tensor input) {
        (void)input;
        errs::forward_not_implemented<Layer>();
    }
    virtual ~Layer() = default;
};

#define FORWARD_ALIASES                                                        \
    Tensor _forward_float(Tensor input) {                                      \
        return forward<float>(std::move(input));                               \
    }                                                                          \
    Tensor _forward_double(Tensor input) {                                     \
        return forward<double>(std::move(input));                              \
    }

class MaxPool2D : virtual public Layer {
  public:
    MaxPool2D(const uint kernel_h, const uint kernel_w, const uint padding_h,
              const uint padding_w, const uint stride_h, const uint stride_w,
              const uint dilation_h, const uint dilation_w,
              const bool ceil_mode, const std::string algorithm)
        : kernel_h(kernel_h), kernel_w(kernel_w), padding_h(padding_h),
          padding_w(padding_w), stride_h(stride_h), stride_w(stride_w),
          dilation_h(dilation_h), dilation_w(dilation_w), ceil_mode(ceil_mode),
          algorithm(algorithm) {}

    FORWARD_ALIASES

    template <typename dtype> Tensor forward(Tensor input) {
        if (algorithm == DEFAULT_OPT_STR) {
            if constexpr (DEFAULT_MAXPOOL2D) {
                return maxpool2d_custom<dtype>(
                    std::move(input), kernel_h, kernel_w, padding_h, padding_w,
                    stride_h, stride_w, dilation_h, dilation_w, ceil_mode);
            } else {
                return maxpool2d::direct<dtype>(
                    std::move(input), kernel_h, kernel_w, padding_h, padding_w,
                    stride_h, stride_w, dilation_h, dilation_w, ceil_mode);
            }
        } else if (algorithm == CUSTOM_OPT_STR) {
            return maxpool2d_custom<dtype>(
                std::move(input), kernel_h, kernel_w, padding_h, padding_w,
                stride_h, stride_w, dilation_h, dilation_w, ceil_mode);
        } else if (algorithm == "direct") {
            return maxpool2d::direct<dtype>(
                std::move(input), kernel_h, kernel_w, padding_h, padding_w,
                stride_h, stride_w, dilation_h, dilation_w, ceil_mode);
        }

        errs::invalid_algo("maxpool2d", algorithm);
    }
    ~MaxPool2D() = default;

  private:
    const uint kernel_h;
    const uint kernel_w;
    const uint padding_h;
    const uint padding_w;
    const uint stride_h;
    const uint stride_w;
    const uint dilation_h;
    const uint dilation_w;
    const bool ceil_mode;
    const std::string algorithm;
};

class AvgPool2D : virtual public Layer {
  public:
    AvgPool2D(const uint kernel_h, const uint kernel_w, const uint padding_h,
              const uint padding_w, const uint stride_h, const uint stride_w,
              const bool ceil_mode, const bool count_include_pad,
              const std::optional<uint> divisor_override,
              const std::string algorithm)
        : kernel_h(kernel_h), kernel_w(kernel_w), padding_h(padding_h),
          padding_w(padding_w), stride_h(stride_h), stride_w(stride_w),
          ceil_mode(ceil_mode), count_include_pad(count_include_pad),
          divisor_override(divisor_override), algorithm(algorithm) {}

    FORWARD_ALIASES

    template <typename dtype> Tensor forward(Tensor input) {
        if (algorithm == DEFAULT_OPT_STR) {
            if constexpr (DEFAULT_AVGPOOL2D) {
                return avgpool2d_custom<dtype>(
                    std::move(input), kernel_h, kernel_w, padding_h, padding_w,
                    stride_h, stride_w, ceil_mode, count_include_pad,
                    divisor_override);
            } else {
                return avgpool2d::direct<dtype>(
                    std::move(input), kernel_h, kernel_w, padding_h, padding_w,
                    stride_h, stride_w, ceil_mode, count_include_pad,
                    divisor_override);
            }
        } else if (algorithm == CUSTOM_OPT_STR) {
            return avgpool2d_custom<dtype>(std::move(input), kernel_h, kernel_w,
                                           padding_h, padding_w, stride_h,
                                           stride_w, ceil_mode,
                                           count_include_pad, divisor_override);
        } else if (algorithm == "direct") {
            return avgpool2d::direct<dtype>(
                std::move(input), kernel_h, kernel_w, padding_h, padding_w,
                stride_h, stride_w, ceil_mode, count_include_pad,
                divisor_override);
        }
        errs::invalid_algo("avgpool2d", algorithm);
    }
    ~AvgPool2D() = default;

  private:
    const uint kernel_h;
    const uint kernel_w;
    const uint padding_h;
    const uint padding_w;
    const uint stride_h;
    const uint stride_w;
    const bool ceil_mode;
    const bool count_include_pad;
    const std::optional<uint> divisor_override;
    const std::string algorithm;
};

class AdaptiveAvgPool2D : virtual public Layer {
  public:
    AdaptiveAvgPool2D(const std::optional<uint> output_h,
                      const std::optional<uint> output_w,
                      const std::string algorithm)
        : output_h(output_h), output_w(output_w), algorithm(algorithm) {}

    FORWARD_ALIASES

    template <typename dtype> Tensor forward(Tensor input) {
        if (algorithm == DEFAULT_OPT_STR) {
            if constexpr (DEFAULT_ADAPTIVEAVGPOOL2D) {
                return adaptiveavgpool2d_custom<dtype>(std::move(input),
                                                       output_h, output_w);
            } else {
                return adaptiveavgpool2d::direct<dtype>(std::move(input),
                                                        output_h, output_w);
            }
        } else if (algorithm == CUSTOM_OPT_STR) {
            return adaptiveavgpool2d_custom<dtype>(std::move(input), output_h,
                                                   output_w);
        } else if (algorithm == "direct") {
            return adaptiveavgpool2d::direct<dtype>(std::move(input), output_h,
                                                    output_w);
        }
        errs::invalid_algo("adaptiveavgpool2d", algorithm);
    }
    ~AdaptiveAvgPool2D() = default;

  private:
    const std::optional<uint> output_h;
    const std::optional<uint> output_w;
    const std::string algorithm;
};

class ReLU : virtual public Layer {
  public:
    ReLU(const std::string algorithm) : algorithm(algorithm){};

    FORWARD_ALIASES

    template <typename dtype> Tensor forward(Tensor input) {
        if (algorithm == DEFAULT_OPT_STR) {
            if constexpr (DEFAULT_RELU) {
                return relu_custom<dtype>(std::move(input));
            } else {
                return relu::direct<dtype>(std::move(input));
            }
        } else if (algorithm == CUSTOM_OPT_STR) {
            return relu_custom<dtype>(std::move(input));
        } else if (algorithm == "direct") {
            return relu::direct<dtype>(std::move(input));
        }

        errs::invalid_algo("relu", algorithm);
    }
    ~ReLU() = default;

  private:
    const std::string algorithm;
};

class Linear : virtual public Layer {
  public:
    Linear(const intptr_t weight_address, const std::vector<uint> weight_shape,
           const std::optional<intptr_t> bias_addr, const std::string algorithm,
           const ScalarType scalar_type)
        : weight(weight_address, weight_shape, scalar_type),
          bias(
              Tensor::from_optional(bias_addr, {weight_shape[0]}, scalar_type)),
          algorithm(algorithm) {}

    FORWARD_ALIASES

    template <typename dtype> Tensor forward(Tensor input) {
        if (algorithm == DEFAULT_OPT_STR) {
            if constexpr (DEFAULT_LINEAR) {
                return linear_custom<dtype>(std::move(input), weight, bias);
            } else {
                return linear::gemm<dtype>(std::move(input), weight, bias);
            }
        } else if (algorithm == CUSTOM_OPT_STR) {
            return linear_custom<dtype>(std::move(input), weight, bias);
        } else if (algorithm == "gemm") {
            return linear::gemm<dtype>(std::move(input), weight, bias);
        }

        errs::invalid_algo("linear", algorithm);
    }

    ~Linear() = default;

  private:
    const Tensor weight;
    const std::optional<const Tensor> bias;
    const std::string algorithm;
};

class Flatten : virtual public Layer {
  public:
    Flatten(const uint start_dim, const int end_dim,
            const std::string algorithm)
        : start_dim(start_dim), end_dim(end_dim), algorithm(algorithm) {}

    FORWARD_ALIASES

    template <typename dtype> Tensor forward(Tensor input) {
        if (algorithm == DEFAULT_OPT_STR) {
            if constexpr (DEFAULT_FLATTEN) {
                return flatten_custom<dtype>(std::move(input), start_dim,
                                             end_dim);
            } else {
                return flatten::direct<dtype>(std::move(input), start_dim,
                                              end_dim);
            }
        } else if (algorithm == CUSTOM_OPT_STR) {
            return flatten_custom<dtype>(std::move(input), start_dim, end_dim);
        } else if (algorithm == "direct") {
            return flatten::direct<dtype>(std::move(input), start_dim, end_dim);
        }

        errs::invalid_algo("flatten", algorithm);
    }
    ~Flatten() = default;

  private:
    const uint start_dim;
    const int end_dim;
    const std::string algorithm;
};

class Conv2D : virtual public Layer {
  public:
    Conv2D(const intptr_t weight_address, const std::vector<uint> weight_shape,
           const std::optional<intptr_t> bias_addr, const uint padding_h,
           const uint padding_w, const uint stride_h, const uint stride_w,
           const uint dilation_h, const uint dilation_w,
           const PaddingMode padding_mode, const uint groups,
           const std::string algorithm, const ScalarType scalar_type,
           bool own_params = true)
        : weight(own_params ? Tensor(weight_address, weight_shape, scalar_type)
                            : Tensor::form_tensor(weight_address, weight_shape,
                                                  scalar_type)),
          bias(Tensor::from_optional(bias_addr, {weight_shape[0]}, scalar_type,
                                     own_params)),
          padding_h(padding_h), padding_w(padding_w), stride_h(stride_h),
          stride_w(stride_w), dilation_h(dilation_h), dilation_w(dilation_w),
          padding_mode(padding_mode), groups(groups), algorithm(algorithm) {}

    FORWARD_ALIASES

    template <typename dtype> Tensor forward(Tensor input) {
        if (algorithm == DEFAULT_OPT_STR) {
            if constexpr (DEFAULT_CONV2D) {
                return conv2d_custom<dtype>(std::move(input), weight, bias,
                                            padding_h, padding_w, stride_h,
                                            stride_w, dilation_h, dilation_w,
                                            padding_mode, groups);
            } else if constexpr (USING_CUDNN) {
                return conv2d::implicit_precomp_gemm<dtype>(
                    std::move(input), weight, bias, padding_h, padding_w,
                    stride_h, stride_w, dilation_h, dilation_w, padding_mode,
                    groups);
            } else if constexpr (USING_MPS_METAL) {
                return conv2d::mps<dtype>(std::move(input), weight, bias,
                                          padding_h, padding_w, stride_h,
                                          stride_w, dilation_h, dilation_w,
                                          padding_mode, groups);
            } else {
                return conv2d::direct<dtype>(std::move(input), weight, bias,
                                             padding_h, padding_w, stride_h,
                                             stride_w, dilation_h, dilation_w,
                                             padding_mode, groups);
            }
        } else if (algorithm == CUSTOM_OPT_STR) {
            return conv2d_custom<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups);
        } else if (algorithm == "mps") {
            return conv2d::mps<dtype>(std::move(input), weight, bias, padding_h,
                                      padding_w, stride_h, stride_w, dilation_h,
                                      dilation_w, padding_mode, groups);
        } else if (algorithm == "metal") {
            return conv2d::metal<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups);
        } else if (algorithm == "direct") {
            return conv2d::direct<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups);
        } else if (algorithm == "smm") {
            return conv2d::smm<dtype>(std::move(input), weight, bias, padding_h,
                                      padding_w, stride_h, stride_w, dilation_h,
                                      dilation_w, padding_mode, groups);
        } else if (algorithm == "implicit precomp gemm") {
            return conv2d::implicit_precomp_gemm<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups);
        } else if (algorithm == "implicit gemm") {
            return conv2d::implicit_gemm<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups);
        } else if (algorithm == "gemm") {
            return conv2d::gemm<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups);
        } else if (algorithm == "winograd") {
            return conv2d::winograd<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups);
        } else if (algorithm == "guess") {
            return conv2d::guess<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups);
        }
        errs::invalid_algo("conv2d", algorithm);
    }

    ~Conv2D() = default;

  private:
    const Tensor weight;
    const std::optional<const Tensor> bias;
    const uint padding_h;
    const uint padding_w;
    const uint stride_h;
    const uint stride_w;
    const uint dilation_h;
    const uint dilation_w;
    const PaddingMode padding_mode;
    const uint groups;
    const std::string algorithm;
};

Tensor conv2d_with_algo(const intptr_t input_address,
                        const std::vector<uint> input_shape,
                        const ScalarType input_type,
                        const intptr_t weight_address,
                        const std::vector<uint> weight_shape,
                        const std::optional<intptr_t> bias_addr,
                        const uint padding_h, const uint padding_w,
                        const uint stride_h, const uint stride_w,
                        const uint dilation_h, const uint dilation_w,
                        const PaddingMode padding_mode, const uint groups,
                        const std::string algorithm) {
    Conv2D layer(weight_address, weight_shape, bias_addr, padding_h, padding_w,
                 stride_h, stride_w, dilation_h, dilation_w, padding_mode,
                 groups, algorithm, input_type, false);

    if (input_type == ScalarType::Float32) {
        return layer.forward<float>(
            Tensor::form_tensor(input_address, input_shape, input_type));
    }
    return layer.forward<double>(
        Tensor::form_tensor(input_address, input_shape, input_type));
}

class MultiheadAttention : virtual public Layer {
  public:
    MultiheadAttention(const intptr_t q_proj_address,
                       const intptr_t k_proj_address,
                       const intptr_t v_proj_address,
                       const std::optional<intptr_t> q_bias_in_address,
                       const std::optional<intptr_t> k_bias_in_address,
                       const std::optional<intptr_t> v_bias_in_address,
                       const std::optional<intptr_t> k_bias_address,
                       const std::optional<intptr_t> v_bias_address,
                       const intptr_t out_proj_address,
                       const std::optional<intptr_t> out_proj_bias_address,
                       const bool add_zero_attn, const uint num_heads,
                       const uint head_dim, const uint k_dim, const uint v_dim,
                       const uint embed_dim, const float dropout,
                       const std::string algorithm,
                       const ScalarType scalar_type, bool own_params = true)
        : q_proj(
              own_params
                  ? Tensor(q_proj_address, {embed_dim, embed_dim}, scalar_type)
                  : Tensor::form_tensor(q_proj_address, {embed_dim, embed_dim},
                                        scalar_type)),
          k_proj(own_params
                     ? Tensor(k_proj_address, {embed_dim, k_dim}, scalar_type)
                     : Tensor::form_tensor(k_proj_address, {embed_dim, k_dim},
                                           scalar_type)),
          v_proj(own_params
                     ? Tensor(v_proj_address, {embed_dim, v_dim}, scalar_type)
                     : Tensor::form_tensor(v_proj_address, {embed_dim, v_dim},
                                           scalar_type)),
          q_bias_in(Tensor::from_optional(q_bias_in_address, {embed_dim},
                                          scalar_type, own_params)),
          k_bias_in(Tensor::from_optional(k_bias_in_address, {embed_dim},
                                          scalar_type, own_params)),
          v_bias_in(Tensor::from_optional(v_bias_in_address, {embed_dim},
                                          scalar_type, own_params)),
          k_bias(Tensor::from_optional(k_bias_address, {1, 1, embed_dim},
                                       scalar_type, own_params)),
          v_bias(Tensor::from_optional(v_bias_address, {1, 1, embed_dim},
                                       scalar_type, own_params)),
          out_proj(own_params ? Tensor(out_proj_address, {embed_dim, embed_dim},
                                       scalar_type)
                              : Tensor::form_tensor(out_proj_address,
                                                    {embed_dim, embed_dim},
                                                    scalar_type)),
          out_bias(Tensor::from_optional(out_proj_bias_address, {embed_dim},
                                         scalar_type, own_params)),
          add_zero_attn(add_zero_attn), num_heads(num_heads),
          head_dim(head_dim), dropout(dropout), algorithm(algorithm) {}

    template <typename dtype>
    Tensor forward(Tensor query, Tensor key, Tensor value,
                   const mha::MemFormat input_format,
                   std::optional<Tensor> attn_mask,
                   std::optional<Tensor> key_padding_mask,
                   const bool need_weights, const bool average_attn_weights,
                   const bool is_causal, const bool need_to_project) {
        if (algorithm == DEFAULT_OPT_STR) {
            if constexpr (DEFAULT_MHA) {
                return mha_custom<dtype>(
                    std::move(query), std::move(key), std::move(value),
                    input_format, q_proj, k_proj, v_proj, q_bias_in, k_bias_in,
                    v_bias_in, k_bias, v_bias, out_proj, out_bias,
                    add_zero_attn, num_heads, head_dim, dropout,
                    key_padding_mask, attn_mask, need_weights,
                    average_attn_weights, is_causal);
            } else {
                return mha::standard<dtype>(
                    std::move(query), std::move(key), std::move(value),
                    input_format, q_proj, k_proj, v_proj, q_bias_in, k_bias_in,
                    v_bias_in, k_bias, v_bias, out_proj, out_bias,
                    add_zero_attn, num_heads, head_dim, dropout,
                    key_padding_mask, attn_mask, need_weights,
                    average_attn_weights, is_causal, need_to_project);
            }
        } else if (algorithm == CUSTOM_OPT_STR) {
            return mha_custom<dtype>(
                std::move(query), std::move(key), std::move(value),
                input_format, q_proj, k_proj, v_proj, q_bias_in, k_bias_in,
                v_bias_in, k_bias, v_bias, out_proj, out_bias, add_zero_attn,
                num_heads, head_dim, dropout, key_padding_mask, attn_mask,
                need_weights, average_attn_weights, is_causal);
        }
        errs::invalid_algo("MultiheadAttention", algorithm);
    }

    ~MultiheadAttention() = default;

  private:
    const Tensor q_proj;
    const Tensor k_proj;
    const Tensor v_proj;
    const std::optional<const Tensor> q_bias_in;
    const std::optional<const Tensor> k_bias_in;
    const std::optional<const Tensor> v_bias_in;
    const std::optional<const Tensor> k_bias;
    const std::optional<const Tensor> v_bias;
    const Tensor out_proj;
    const std::optional<const Tensor> out_bias;
    const bool add_zero_attn;
    const std::optional<const Tensor> key_padding_mask;
    const std::optional<const Tensor> attn_mask;
    const uint num_heads;
    const uint head_dim;
    const float dropout;
    const std::string algorithm;
};

// TODO when doing the weights will have to make this maybe return two
// things
Tensor
mha_with_algo(const intptr_t q_address, const intptr_t k_address,
              const intptr_t v_address, const ScalarType input_type,
              const mha::MemFormat input_format,
              const std::vector<uint> q_shape, const std::vector<uint> k_shape,
              const std::vector<uint> v_shape, const intptr_t q_proj_address,
              const intptr_t k_proj_address, const intptr_t v_proj_address,
              const std::optional<intptr_t> q_bias_in_address,
              const std::optional<intptr_t> k_bias_in_address,
              const std::optional<intptr_t> v_bias_in_address,
              const std::optional<intptr_t> k_bias_address,
              const std::optional<intptr_t> v_bias_address,
              const intptr_t out_proj_address,
              const std::optional<intptr_t> out_proj_bias_address,
              const bool add_zero_attn, const uint num_heads,
              const uint head_dim, const uint k_dim, const uint v_dim,
              const uint embed_dim, const float dropout,
              const std::optional<intptr_t> attn_mask_address,
              const std::optional<intptr_t> key_padding_mask_address,
              const bool need_weights, const bool average_attn_weights,
              const bool is_causal, const bool need_to_project,
              const std::string algorithm) {
    MultiheadAttention layer(q_proj_address, k_proj_address, v_proj_address,
                             q_bias_in_address, k_bias_in_address,
                             v_bias_in_address, k_bias_address, v_bias_address,
                             out_proj_address, out_proj_bias_address,
                             add_zero_attn, num_heads, head_dim, k_dim, v_dim,
                             embed_dim, dropout, algorithm, input_type, false);
    std::optional<Tensor> attn_mask =
        attn_mask_address
            ? std::optional<Tensor>(Tensor::form_tensor(
                  *attn_mask_address,
                  {q_shape[q_shape.size() - 2], k_shape[k_shape.size() - 2]},
                  input_type))
            : std::nullopt;
    std::optional<Tensor> key_padding_mask =
        key_padding_mask_address
            ? std::optional<Tensor>(Tensor::form_tensor(
                  *key_padding_mask_address,
                  k_shape.size() == 4
                      ? std::vector<uint>{k_shape[0],
                                          k_shape[k_shape.size() - 2]}
                      : std::vector<uint>{k_shape[k_shape.size() - 2]},
                  input_type))
            : std::nullopt;

    std::cout << "v shape" << std::endl;
    for (auto &s : v_shape) {
        std::cout << s << std::endl;
    }
    std::cout << "end" << std::endl;

    if (input_type == ScalarType::Float32) {
        return layer.forward<float>(
            Tensor::form_tensor(q_address, q_shape, input_type),
            Tensor::form_tensor(k_address, k_shape, input_type),
            Tensor::form_tensor(v_address, v_shape, input_type), input_format,
            std::move(attn_mask), std::move(key_padding_mask), need_weights,
            average_attn_weights, is_causal, need_to_project);
    }
    return layer.forward<double>(
        Tensor::form_tensor(q_address, q_shape, input_type),
        Tensor::form_tensor(k_address, k_shape, input_type),
        Tensor::form_tensor(v_address, v_shape, input_type), input_format,
        std::move(attn_mask), std::move(key_padding_mask), need_weights,
        average_attn_weights, is_causal, need_to_project);
}

class Model {
  public:
    Model(const std::vector<std::shared_ptr<Layer>> layers) : layers(layers) {}

    Tensor predict(const intptr_t input_address, std::vector<uint> input_shape,
                   ScalarType input_type) {
        Tensor output =
            Tensor::form_tensor(input_address, input_shape, input_type);
        for (const std::shared_ptr<Layer> &layer : layers) {
            if (input_type == ScalarType::Float32) {
                output = layer->_forward_float(std::move(output));
            } else {
                output = layer->_forward_double(std::move(output));
            }
        }
        return output;
    }

    ~Model() = default;

  private:
    const std::vector<std::shared_ptr<Layer>> layers;
};

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    py::class_<Tensor>(m, "Tensor", pybind11::buffer_protocol())
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("scalar_type", &Tensor::scalar_type)
        .def_buffer(&Tensor::buffer);

    py::class_<Model>(m, "Model")
        .def(py::init<const std::vector<std::shared_ptr<Layer>>>())
        .def("predict", &Model::predict);

    py::class_<Layer, std::shared_ptr<Layer>> _(m, "Layer");

    py::class_<MaxPool2D, Layer, std::shared_ptr<MaxPool2D>>(m, "MaxPool2D")
        .def(py::init<const uint, const uint, const uint, const uint,
                      const uint, const uint, const uint, const uint,
                      const bool, const std::string>());

    py::class_<Conv2D, Layer, std::shared_ptr<Conv2D>>(m, "Conv2D")
        .def(py::init<const intptr_t, const std::vector<uint>,
                      const std::optional<intptr_t>, const uint, const uint,
                      const uint, const uint, const uint, const uint,
                      const PaddingMode, const uint, const std::string,
                      const ScalarType>());

    m.def("conv2d", &conv2d_with_algo);
    m.def("mha", &mha_with_algo);

    py::class_<Linear, Layer, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<const intptr_t, const std::vector<uint>,
                      const std::optional<intptr_t>, const std::string,
                      const ScalarType>());

    py::class_<ReLU, Layer, std::shared_ptr<ReLU>>(m, "ReLU").def(
        py::init<const std::string>());

    py::class_<AvgPool2D, Layer, std::shared_ptr<AvgPool2D>>(m, "AvgPool2D")
        .def(py::init<const uint, const uint, const uint, const uint,
                      const uint, const uint, const bool, const bool,
                      const std::optional<uint>, const std::string>());

    py::class_<AdaptiveAvgPool2D, Layer, std::shared_ptr<AdaptiveAvgPool2D>>(
        m, "AdaptiveAvgPool2D")
        .def(py::init<const std::optional<uint>, const std::optional<uint>,
                      const std::string>());

    py::class_<Flatten, Layer, std::shared_ptr<Flatten>>(m, "Flatten")
        .def(py::init<const uint, const int, const std::string>());

    py::enum_<PaddingMode>(m, "PaddingMode")
        .value("zeros", PaddingMode::Zeros)
        .value("reflect", PaddingMode::Reflect)
        .value("replicate", PaddingMode::Replicate)
        .value("circular", PaddingMode::Circular)
        .export_values();
    py::enum_<mha::MemFormat>(m, "MHAMemFormat")
        .value("NSE", mha::MemFormat::NSE)
        .value("SNE", mha::MemFormat::SNE)
        .export_values();
    py::enum_<ScalarType>(m, "ScalarType")
        .value("Float32", ScalarType::Float32)
        .value("Float64", ScalarType::Float64)
        .export_values();
    m.def("output_hw_for_2d", &output_hw_for_2d_no_ceil);
    m.def("using_mps_and_metal", [] { return USING_MPS_METAL; });
    m.def("using_sycl", [] { return USING_SYCL; });
    m.def("using_cublas", [] { return USING_CUBLAS; });
    m.def("using_cudnn", [] { return USING_CUDNN; });
    m.def("default_opt_str", [] { return DEFAULT_OPT_STR; });
    m.def("custom_opt_str", [] { return CUSTOM_OPT_STR; });
    m.def("custom_mha_handles_inputs", [] {
        return CUSTOM_MHA_HANDLES_INPUTS.has_value() &&
               *CUSTOM_MHA_HANDLES_INPUTS;
    });
    m.def("custom_mha_project_output", [] {
        return CUSTOM_MHA_PROJECTS_OUTPUT.has_value() &&
               *CUSTOM_MHA_PROJECTS_OUTPUT;
    });

    static_assert(sizeof(float) == 4,
                  "expected `float` to be 4 bytes (float32)");
    static_assert(sizeof(double) == 8,
                  "expected `double` to be 8 bytes (float64)");
}
