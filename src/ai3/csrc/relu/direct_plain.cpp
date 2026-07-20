// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>

template <typename dtype> Tensor relu::direct(Tensor input) {
    int total_elements = input.count();
    dtype *in_data = data_as<dtype>(input.data);
    for (int i = 0; i < total_elements; i++) {
        in_data[i] = (in_data[i] > 0) ? in_data[i] : 0;
    }
    return input;
}

template Tensor relu::direct<float>(RELU_PARAMS);
template Tensor relu::direct<double>(RELU_PARAMS);
