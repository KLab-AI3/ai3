// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "utils.hpp"
#include <algorithm>
#include <cstring>
#include <optional>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

/**
 * @brief Used for built-in and custom implementations of common deep learning
 * operations
 *
 * Represents a multi-dimensional tensor which can be initialized from raw data
 * or constructed from existing data with optional ownership semantics.
 *
 * @tparam dtype The data type of the tensor elements
 */
class Tensor {
  public:
    /**
     * @brief Allocates data to construct a Tensor with the given shape
     *
     * @param s Shape of the tensor
     * @param scalar_type type of data to store
     */
    Tensor(const std::vector<uint> &s, ScalarType scalar_type)
        : data(new_data_with_type(scalar_type, _count(s))), shape(s),
          owned(true), scalar_type(scalar_type) {}

    /**
     * @brief Allocates and copies data to construct a Tensor with the given
     * shape and data
     *
     * @param data_address Address of the data to be copied to the Tensor
     * @param s Shape of the tensor
     * @param scalar_type type of the `data_address`
     */
    Tensor(const intptr_t data_address, const std::vector<uint> &s,
           ScalarType scalar_type)
        : shape(s), owned(true), scalar_type(scalar_type) {
        data = new_data_with_type(scalar_type, _count(s));
        std::memcpy(data, reinterpret_cast<const void *>(data_address),
                    _count(s) * size_of_scalar_type(scalar_type));
    }

    /**
     * @brief Creates a tensor with the given values
     *
     * @param data Address of the data
     * @param s Shape of the tensor
     * @param own Whether the Tensor owns this data
     * @param scalar_type type of the `data`
     */
    Tensor(void *data, const std::vector<uint> &s, bool own,
           ScalarType scalar_type)
        : data(data), shape(s), owned(own), scalar_type(scalar_type) {}

    /**
     * @brief Wraps existing data with a Tensor without allocating or copying
     *
     * @param data_address Address of the data
     * @param s Shape of the tensor
     * @param scalar_type type of the `data_address`
     *
     * @return A Tensor object around the provided data
     */
    static Tensor form_tensor(const intptr_t data_address,
                              const std::vector<uint> &s,
                              ScalarType scalar_type) {
        return Tensor(reinterpret_cast<void *>(data_address), s, false,
                      scalar_type);
    }

    /**
     * @brief Creates a Tensor object depending on an optional data
     * address
     *
     * If the data address is present, a Tensor is created with ownership
     * determined by the *own* parameter. If the data address is not present,
     * *std::nullopt* is returned.
     *
     * @param data_address Optional address of the raw data.
     * @param s Shape of the tensor.
     * @param scalar_type type of the `data_address`
     * @param own Whether to take ownership of the data.
     *
     * @return Tensor object if data_address has a value; otherwise,
     * *std::nullopt*.
     */
    static std::optional<Tensor>
    from_optional(const std::optional<intptr_t> &data_address,
                  const std::vector<uint> &s, ScalarType scalar_type,
                  bool own = true) {
        if (data_address.has_value()) {
            if (own) {
                return Tensor(*data_address, s, scalar_type);
            }
            return form_tensor(*data_address, s, scalar_type);
        }
        return std::nullopt;
    }

    /**
     * @brief Creates a new tensor from the exsiting with data of specified type
     *
     * @tparam target_type The type to which the data should be converted.
     *
     * @return Tensor containing the converted data
     */
    template <typename target_type> Tensor to_type(ScalarType to_type) const {
        target_type *new_data = new target_type[count()];
        if (scalar_type == ScalarType::Float32) {
            copy_with_types<target_type, float>(new_data, data, count());
        } else {
            copy_with_types<target_type, double>(new_data, data, count());
        }
        return Tensor(new_data, shape, true, to_type);
    }

    template <typename target_type, typename orig_type>
    void copy_with_types(target_type *new_data, void *_orig_data,
                         uint count) const {
        orig_type *orig_data = (orig_type *)_orig_data;
        for (uint i = 0; i < count; ++i) {
            new_data[i] = orig_data[i];
        }
    }

    Tensor() = default;

    ~Tensor() {
        if (owned) {
            if (scalar_type == ScalarType::Float32) {
                delete[] (float *)data;
            } else {
                delete[] (double *)data;
            }
        }
    };

    Tensor(Tensor &&other) noexcept { *this = std::move(other); }

    Tensor &operator=(Tensor &&other) noexcept {
        if (this != &other) {
            shape = std::move(other.shape);
            data = other.data;
            owned = other.owned;
            scalar_type = other.scalar_type;
            other.data = nullptr;
            other.owned = false;
        }
        return *this;
    }

    /**
     * @brief Implementation of
     * <a href="https://docs.python.org/3/c-api/buffer.html">Python Buffer
     * Protocol</a> for interoperability with Python.
     *
     * @return *pybind11::buffer_info* object containing the tensor's data,
     * shape, and strides.
     */
    py::buffer_info buffer() {
        std::vector<uint> stride(shape.size());
        stride[shape.size() - 1] = size_of_scalar_type(scalar_type);
        for (int i = shape.size() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        std::string format;
        if (scalar_type == ScalarType::Float32) {
            format = py::format_descriptor<float>::format();
        } else {
            format = py::format_descriptor<double>::format();
        }
        return py::buffer_info(data, size_of_scalar_type(scalar_type), format,
                               shape.size(), shape, stride);
    }

    /**
     * @param data_dim The number of dimensions required per sample, see
     * `sample_dims`
     *
     * @return *true* if the tensor has a dimension for batch size false
     * otherwise
     */
    inline bool batched(const int data_dim) const {
        return shape.size() == unsigned(data_dim + 1);
    }

    /**
     * @param input_dims The number of dimensions required per sample, see
     * `sample_dims`
     *
     * @return the number of samples in a batched Tensor
     */
    inline uint batch_size(const uint input_dims) const {
        return shape[shape.size() - 1 - input_dims];
    }

    /**
     * @return Number of output channels of the Tensor
     */
    inline uint output_channels() const { return shape[shape.size() - 4]; }
    /**
     * @return Number of input channels of the Tensor
     */
    inline uint input_channels() const { return shape[shape.size() - 3]; }
    /**
     * @return Height of the Tensor
     */
    inline uint height() const { return shape[shape.size() - 2]; }
    /**
     * @return Width of the Tensor
     */
    inline uint width() const { return shape[shape.size() - 1]; }

    /**
     * @return Number of elements in the Tensor
     */
    inline uint count() const { return _count(shape); }

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    void *data;              ///< Pointer to the tensor data
    std::vector<uint> shape; ///< Shape of the tensor
    bool owned;              ///< Indicates whether the tensor owns @ref data
    ScalarType scalar_type;  ///< The type stored in @ref data

  private:
    static uint _count(const std::vector<uint> &s) {
        if (s.empty()) {
            return 0;
        }
        uint count = 1;
        for (uint v : s) {
            count *= v;
        }
        return count;
    }
};

inline void ensure_same_type(const Tensor &a, const Tensor &b) {
    errs::bail_if(a.scalar_type != b.scalar_type,
                  "tensors have different data types");
}
inline void ensure_same_type(const Tensor &a, const Tensor &b,
                             const std::optional<const Tensor> &c) {
    ensure_same_type(a, b);
    if (c.has_value()) {
        ensure_same_type(a, *c);
    }
}
