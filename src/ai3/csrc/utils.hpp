// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>

#ifndef GROUP_SIZE_GUESS
#define GROUP_SIZE_GUESS 1024
#endif

enum class PaddingMode { Zeros, Reflect, Replicate, Circular };

enum class ScalarType {
    Float32,
    Float64,
};

template <typename dtype> inline dtype *data_as(void *data) {
    return reinterpret_cast<dtype *>(data);
}

inline std::ostream &operator<<(std::ostream &os, ScalarType type) {
    std::string rep;
    switch (type) {
    case ScalarType::Float32:
        rep = "Float";
    case ScalarType::Float64:
        rep = "Double";
    }
    return os << rep;
}

struct GroupSplit2D {
    uint each_a;
    uint each_b;
    uint total_a;
    uint total_b;
};

template <typename dtype>
void proportionate_2d_work_split(uint num_a, uint num_b, uint total,
                                 uint *set_each_a, uint *set_each_b,
                                 uint *set_total_a, uint *set_total_b) {
    dtype scaler = std::sqrt(dtype(total) / (num_a * num_b));
    uint each_a = num_a * scaler;
    uint each_b = num_b * scaler;
    if (each_a == 0) {
        each_a = 1;
        scaler = 1 / (scaler * num_a);
        each_b /= scaler;
        if (each_b == 0) {
            each_b = 1;
        }
    }
    if (each_b == 0) {
        each_b = 1;
        scaler = 1 / (scaler * num_b);
        each_a /= scaler;
        if (each_a == 0) {
            each_a = 1;
        }
    }
    *set_each_a = each_a;
    *set_each_b = each_b;
    *set_total_a = ((num_a + each_a - 1) / each_a) * each_a;
    *set_total_b = ((num_b + each_b - 1) / each_b) * each_b;
}

inline uint to_linear(uint i, uint j, uint k, uint l, uint m, uint J, uint K,
                      uint L, uint M) {
    return (((i * J + j) * K + k) * L + l) * M + m;
}
inline uint to_linear(uint i, uint j, uint k, uint l, uint m, uint n, uint J,
                      uint K, uint L, uint M, uint N) {
    return ((((i * J + j) * K + k) * L + l) * M + m) * N + n;
}
inline uint to_linear(uint i, uint j, uint k, uint l, uint J, uint K, uint L) {
    return ((i * J + j) * K + k) * L + l;
}
inline uint to_linear(uint i, uint j, uint k, uint J, uint K) {
    return (i * J + j) * K + k;
}
inline uint to_linear(uint i, uint j, uint J) { return i * J + j; }

inline uint output_hw_for_2d_numerator(const uint input, const uint kernel,
                                       const uint padding,
                                       const std::optional<uint> dilation) {
    return input + (2 * padding) - (dilation.value_or(1) * (kernel - 1)) - 1;
}

inline uint output_hw_for_2d_no_ceil(const uint input, const uint kernel,
                                     const uint padding,
                                     const std::optional<uint> dilation,
                                     const uint stride) {
    return output_hw_for_2d_numerator(input, kernel, padding, dilation) /
               stride +
           1;
}

template <typename dtype>
uint output_hw_for_2d(const uint input, const uint kernel, const uint padding,
                      const std::optional<uint> dilation, const uint stride,
                      const bool ceil_mode) {
    const uint top =
        output_hw_for_2d_numerator(input, kernel, padding, dilation);
    const uint bot = stride;

    const uint poss =
        (ceil_mode ? static_cast<uint>(std::ceil(static_cast<dtype>(top) / bot))
                   : top / bot) +
        1;

    if (!ceil_mode) {
        return poss;
    }

    return (poss - 1) * stride >= input + padding ? poss - 1 : poss;
}

namespace errs {
template <typename... Args> [[noreturn]] void bail(Args... args) {
    std::stringstream ss;
    (ss << ... << args);
    throw std::runtime_error(ss.str());
}

[[noreturn]] inline void invalid_dtype(const ScalarType type) {
    errs::bail("invalid dtype: ", type);
}

[[noreturn]] inline void no_user_def(const std::string &name) {
    bail("trying to use user ", name, " when no implementation exists");
}

[[noreturn]] inline void invalid_algo(const std::string &op,
                                      const std::string &algo) {
    bail("invalid ", op, " algorithm: ", algo);
}

[[noreturn]] inline void invalid_context_access(const std::string &thing,
                                                const std::string &platform) {
    bail("trying to get ", thing, " when ", platform, " is not supported");
}

template <typename T> [[noreturn]] inline void forward_not_implemented() {
    bail("forward not implemented for: " + std::string(typeid(T).name()));
}

inline void warning(const std::string msg) {
    std::cerr << "warning: " << msg << std::endl;
}

inline void mps_metal_unsupported_double() {
    warning("MPS/metal does not support double precision, transforming tensors "
            "to "
            "float "
            "precision and back see: "
            "https://developer.apple.com/documentation/metalperformanceshaders/"
            "mpsdatatype");
}

template <typename... Args> void bail_if(bool check, Args... args) {
    if (check) {
        bail(args...);
    }
}
} // namespace errs

inline void *new_data_with_type(const ScalarType type, const uint count) {
    switch (type) {
    case ScalarType::Float32:
        return new float[count];
    case ScalarType::Float64:
        return new double[count];
    default:
        errs::invalid_dtype(type);
    }
}

inline uint prev_power_of_two(uint x) {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x - (x >> 1);
}

inline uint size_of_scalar_type(ScalarType type) {
    switch (type) {
    case ScalarType::Float32:
        return sizeof(float);
    case ScalarType::Float64:
        return sizeof(double);
    default:
        errs::invalid_dtype(type);
    }
}

/**
 * @brief Number of dimensions per sample of common deep learning operations.
 */
namespace sample_dims {
const int LINEAR = 1;
const int POOL2D = 3;
const int CONV2D = 3;
const int MHA_NOT_PROJECTED = 3;
const int MHA_PROJECTED = 4;
const int ACTIVATION = -1;
const int FLATTEN = -1;
}; // namespace sample_dims
