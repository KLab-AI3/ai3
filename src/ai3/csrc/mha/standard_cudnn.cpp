#include <ai3.hpp>
#include <algos.hpp>
#include <cuda_runtime.h>
#include <cuda_utils.hpp>
#include <cudnn.h>
#include <cudnn_utils.hpp>

const int WEIGHT_RANK = 3;

// TODO do a kernel for reordering
template <typename dtype>
void init_weights(cudnnHandle_t handle, cudnnAttnDescriptor_t attn_desc,
                  cudnnMultiHeadAttnWeightKind_t kind, bool is_proj_weights,
                  uint num_weights, size_t size_weights,
                  cudnnTensorDescriptor_t desc, void *dev_w, void *host_data,
                  uint num_heads, uint head_dim, uint embed_dim, bool identity,
                  cudaStream_t &stream) {
    int dim[WEIGHT_RANK], stride[WEIGHT_RANK];
    int ndim;
    dtype *weight_addr = nullptr;

    CUDNN_CHECK(cudnnGetMultiHeadAttnWeights(handle, attn_desc, kind,
                                             size_weights, dev_w, desc,
                                             (void **)&weight_addr));

    cudnnDataType_t data_type_unused;
    CUDNN_CHECK(cudnnGetTensorNdDescriptor(desc, WEIGHT_RANK, &data_type_unused,
                                           &ndim, dim, stride));

    assert(ndim == WEIGHT_RANK);
    if (is_proj_weights) {
        dtype *reordered_weights = new dtype[num_weights];
        if (identity) {
            std::fill(reordered_weights, reordered_weights + num_weights, 0);
            uint min_dim = std::min(embed_dim, num_heads * head_dim);
            for (uint i = 0; i < min_dim; ++i) {
                reordered_weights[i * num_heads * head_dim + i] =
                    1; // TODO cudaMemset2D
            }
        } else {
            // TODO use the cudaMemcpy2DAsync function, I think it can just
            // be one call don't need to do this reordering here
            for (uint e = 0; e < embed_dim; ++e) {
                for (uint d = 0; d < head_dim * num_heads; ++d) {
                    reordered_weights[e * head_dim * num_heads + d] =
                        ((dtype *)host_data)[d * embed_dim + e];
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(weight_addr, reordered_weights,
                              num_weights * sizeof(dtype),
                              cudaMemcpyHostToDevice));
        delete[] reordered_weights;
    } else {
        if (identity) {
            CUDA_CHECK(cudaMemsetAsync(weight_addr, 0,
                                       num_weights * sizeof(dtype), stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(weight_addr, host_data,
                                       num_weights * sizeof(dtype),
                                       cudaMemcpyHostToDevice, stream));
        }
    }
}

template <typename dtype>
Tensor mha::standard(Tensor query, Tensor key, Tensor value,
                     const mha::MemFormat input_format, const Tensor &q_proj,
                     const Tensor &k_proj, const Tensor &v_proj,
                     const std::optional<const Tensor> &q_bias_in,
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
                     const bool is_causal, const bool need_to_project_input) {
    ensure_same_type(query, key, value);
    errs::bail_if(need_weights, "no support for attention weights");
    errs::bail_if(need_weights and average_attn_weights,
                  "no support for average attention weights");
    errs::bail_if(key_padding_mask.has_value(),
                  "no support for key padding mask");
    errs::bail_if(attn_mask.has_value(), "no support for attention mask");
    errs::bail_if(is_causal, "no support for causal");

    uint batch_size, seq_len_q, embed_q, seq_len_k, embed_k, embed_v, proj_q,
        proj_k, proj_v, embed_o, proj_o;
    if (need_to_project_input) {
        errs::bail_if(query.shape.size() != sample_dims::MHA_NOT_PROJECTED,
                      "need to project in mha but rank of input is: ",
                      query.shape.size());

        uint seq_len_dim, batch_size_dim;
        if (input_format == mha::MemFormat::NSE) {
            batch_size_dim = 0;
            seq_len_dim = 1;
        } else if (input_format == mha::MemFormat::SNE) {
            seq_len_dim = 0;
            batch_size_dim = 1;
        }
        batch_size = query.shape[batch_size_dim];
        seq_len_q = query.shape[seq_len_dim];
        seq_len_k = key.shape[seq_len_dim];
        embed_q = query.shape[2];
        embed_k = key.shape[2];
        embed_v = value.shape[2];
        proj_q = embed_q / num_heads;
        proj_k = embed_q / num_heads;
        proj_v = embed_q / num_heads;
        embed_o = embed_q;
        proj_o = embed_q;
    } else {
        errs::bail_if(
            query.shape.size() != sample_dims::MHA_PROJECTED,
            "input is projected but rank of input is: ", query.shape.size());

        uint seq_len_dim, batch_size_dim;
        if (input_format == mha::MemFormat::NSHD) {
            batch_size_dim = 0;
            seq_len_dim = 1;
        } else if (input_format == mha::MemFormat::SNHD) {
            seq_len_dim = 0;
            batch_size_dim = 1;
        }
        batch_size = query.shape[batch_size_dim];
        seq_len_q = query.shape[seq_len_dim];
        seq_len_k = key.shape[seq_len_dim];
        proj_q = query.shape[3];
        proj_k = key.shape[3];
        proj_v = value.shape[3];
        embed_q = proj_q * num_heads;
        embed_k = proj_k * num_heads;
        embed_v = proj_v * num_heads;
        embed_o = embed_q;
        proj_o = embed_q;
    }
    const double sm_scaler = 1.0 / std::sqrt(embed_q / num_heads);

    const bool proj_bias = q_bias_in.has_value() && k_bias_in.has_value() &&
                           v_bias_in.has_value() && out_bias.has_value();

    std::vector<uint> o_shape(sample_dims::MHA_NOT_PROJECTED);
    if (input_format == mha::MemFormat::NSE ||
        input_format == mha::MemFormat::NSHD) {
        o_shape = {batch_size, seq_len_q, embed_o};
    } else if (input_format == mha::MemFormat::SNE ||
               input_format == mha::MemFormat::SNHD) {
        o_shape = {seq_len_q, batch_size, embed_o};
    }
    Tensor output(std::move(o_shape), query.scalar_type);

    cudnnHandle_t handle = (cudnnHandle_t)Context::cudnn_handle_t();
    cudaStream_t weight_stream;
    cudaStream_t data_stream;
    cudnnAttnDescriptor_t attn_desc;
    cudnnDropoutDescriptor_t drop_desc;
    cudnnSeqDataDescriptor_t q_desc;
    cudnnSeqDataDescriptor_t k_desc;
    cudnnSeqDataDescriptor_t v_desc;
    cudnnSeqDataDescriptor_t o_desc;
    cudnnDataType_t data_type = cudnn_data_type<dtype>();
    cudnnDataType_t comp_prec = cudnn_data_type<dtype>();

    CUDNN_CHECK(cudnnCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&weight_stream));
    CUDA_CHECK(cudaStreamCreate(&data_stream));
    CUDNN_CHECK(cudnnCreateAttnDescriptor(&attn_desc));
    if (dropout > 0) {
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&drop_desc));
    }
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&q_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&k_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&v_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&o_desc));

    uint *loWinIdx = nullptr;
    uint *hiWinIdx = nullptr;
    unsigned attn_mode = proj_bias ? CUDNN_ATTN_ENABLE_PROJ_BIASES
                                   : CUDNN_ATTN_DISABLE_PROJ_BIASES;

    uint qo_tokens = seq_len_q * batch_size;
    uint kv_tokens = seq_len_k * batch_size;

    uint q_num_weights = embed_q * proj_q * num_heads;
    uint k_num_weights = embed_k * proj_k * num_heads;
    uint v_num_weights = embed_v * proj_v * num_heads;
    uint o_num_weights = embed_o * proj_o;

    uint q_bias_len = proj_q * num_heads;
    uint k_bias_len = proj_k * num_heads;
    uint v_bias_len = proj_v * num_heads;
    uint o_bias_len = proj_o;

    uint q_num_elem = qo_tokens * embed_q;
    uint k_num_elem = kv_tokens * embed_k;
    uint v_num_elem = kv_tokens * embed_v;
    uint o_num_elem = qo_tokens * embed_o;

    dtype *dev_q = nullptr;
    dtype *dev_k = nullptr;
    dtype *dev_v = nullptr;
    dtype *dev_o = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&dev_q, q_num_elem * sizeof(dtype)));
    CUDA_CHECK(cudaMalloc((void **)&dev_k, k_num_elem * sizeof(dtype)));
    CUDA_CHECK(cudaMalloc((void **)&dev_v, v_num_elem * sizeof(dtype)));
    CUDA_CHECK(cudaMalloc((void **)&dev_o, o_num_elem * sizeof(dtype)));

    size_t dropout_buf_size;
    void *dropout_buf = nullptr;
    if (dropout > 0) {
        CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &dropout_buf_size));
        CUDA_CHECK(cudaMalloc(&dropout_buf, dropout_buf_size));
        CUDNN_CHECK(cudnnSetDropoutDescriptor(
            drop_desc, handle, dropout, dropout_buf, dropout_buf_size, 0));
    }
    CUDNN_CHECK(cudnnSetAttnDescriptor(
        attn_desc, attn_mode, num_heads, sm_scaler, data_type, comp_prec,
        CUDNN_DEFAULT_MATH, dropout > 0 ? drop_desc : nullptr, nullptr, embed_q,
        embed_k, embed_v, proj_q, proj_k, proj_v, proj_o, seq_len_q, seq_len_k,
        batch_size, 1));

    size_t size_weights = 0, size_wkspace = 0;
    dtype *dev_w = nullptr;
    dtype *dev_wkspace = nullptr;
    CUDNN_CHECK(cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &size_weights,
                                             &size_wkspace, nullptr));
    uint total_num_weights =
        o_num_weights + q_num_weights + k_num_weights + v_num_weights;
    if (proj_bias) {
        total_num_weights += q_bias_len + k_bias_len + v_bias_len + o_bias_len;
    }
    assert(size_weights / sizeof(dtype) == total_num_weights);

    cudnnTensorDescriptor_t weight_desc;
    if (size_weights > 0) {
        CUDA_CHECK(cudaMalloc((void **)&dev_w, size_weights));
        // TODO after alloc weights if don't need to project
        // then memsetasync then sync before init
        // init just writes the ones where needed
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&weight_desc));
        init_weights<dtype>(handle, attn_desc, CUDNN_MH_ATTN_Q_WEIGHTS, true,
                            q_num_weights, size_weights, weight_desc, dev_w,
                            q_proj.data, num_heads, proj_q, embed_q,
                            !need_to_project_input, weight_stream);
        init_weights<dtype>(handle, attn_desc, CUDNN_MH_ATTN_K_WEIGHTS, true,
                            k_num_weights, size_weights, weight_desc, dev_w,
                            k_proj.data, num_heads, proj_k, embed_k,
                            !need_to_project_input, weight_stream);
        init_weights<dtype>(handle, attn_desc, CUDNN_MH_ATTN_V_WEIGHTS, true,
                            v_num_weights, size_weights, weight_desc, dev_w,
                            v_proj.data, num_heads, proj_v, embed_v,
                            !need_to_project_input, weight_stream);
        init_weights<dtype>(handle, attn_desc, CUDNN_MH_ATTN_O_WEIGHTS, true,
                            o_num_weights, size_weights, weight_desc, dev_w,
                            out_proj.data, 1, proj_o, embed_o, false,
                            weight_stream);
        if (proj_bias) {
            init_weights<dtype>(handle, attn_desc, CUDNN_MH_ATTN_Q_BIASES,
                                false, q_bias_len, size_weights, weight_desc,
                                dev_w, q_bias_in->data, 0, 0, 0,
                                !need_to_project_input, weight_stream);
            init_weights<dtype>(handle, attn_desc, CUDNN_MH_ATTN_K_BIASES,
                                false, k_bias_len, size_weights, weight_desc,
                                dev_w, k_bias_in->data, 0, 0, 0,
                                !need_to_project_input, weight_stream);
            init_weights<dtype>(handle, attn_desc, CUDNN_MH_ATTN_V_BIASES,
                                false, v_bias_len, size_weights, weight_desc,
                                dev_w, v_bias_in->data, 0, 0, 0,
                                !need_to_project_input, weight_stream);
            init_weights<dtype>(handle, attn_desc, CUDNN_MH_ATTN_O_BIASES,
                                false, o_bias_len, size_weights, weight_desc,
                                dev_w, out_bias->data, 0, 0, 0, false,
                                weight_stream);
        }
    }
    if (size_wkspace > 0) {
        CUDA_CHECK(cudaMalloc((void **)&dev_wkspace, size_wkspace));
    }

    int *q_seq_array = new int[batch_size]();
    std::fill(q_seq_array, q_seq_array + batch_size, seq_len_q);
    int *k_seq_array = new int[batch_size]();
    std::fill(k_seq_array, k_seq_array + batch_size, seq_len_k);

    int *lo_win_idx = new int[seq_len_q]();
    int *hi_win_idx = new int[seq_len_q];
    std::fill(hi_win_idx, hi_win_idx + seq_len_q, seq_len_k);

    int *dev_q_seq_array = nullptr;
    int *dev_k_seq_array = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&dev_q_seq_array, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(dev_q_seq_array, q_seq_array,
                               batch_size * sizeof(int), cudaMemcpyHostToDevice,
                               data_stream));

    CUDA_CHECK(cudaMalloc((void **)&dev_k_seq_array, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(dev_k_seq_array, k_seq_array,
                               batch_size * sizeof(int), cudaMemcpyHostToDevice,
                               data_stream));

    int dim_a[CUDNN_SEQDATA_DIM_COUNT];
    cudnnSeqDataAxis_t data_axes[CUDNN_SEQDATA_DIM_COUNT];
    data_axes[0] = CUDNN_SEQDATA_BEAM_DIM;
    uint batch_dim, seq_dim;
    if (input_format == mha::MemFormat::NSE ||
        input_format == mha::MemFormat::NSHD) {
        batch_dim = 1;
        seq_dim = 2;
    } else if (input_format == mha::MemFormat::SNE ||
               input_format == mha::MemFormat::SNHD) {
        seq_dim = 1;
        batch_dim = 2;
    }
    data_axes[batch_dim] = CUDNN_SEQDATA_BATCH_DIM;
    data_axes[seq_dim] = CUDNN_SEQDATA_TIME_DIM;
    data_axes[3] = CUDNN_SEQDATA_VECT_DIM;

    dim_a[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dim_a[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
    dim_a[CUDNN_SEQDATA_TIME_DIM] = seq_len_q;

    dim_a[CUDNN_SEQDATA_VECT_DIM] = embed_q;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        q_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a, data_axes,
        batch_size, q_seq_array, nullptr));

    dim_a[CUDNN_SEQDATA_VECT_DIM] = embed_o;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        o_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a, data_axes,
        batch_size, q_seq_array, nullptr));

    dim_a[CUDNN_SEQDATA_TIME_DIM] = seq_len_k;
    dim_a[CUDNN_SEQDATA_VECT_DIM] = embed_k;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        k_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a, data_axes,
        batch_size, k_seq_array, nullptr));

    dim_a[CUDNN_SEQDATA_VECT_DIM] = embed_v;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        v_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a, data_axes,
        batch_size, k_seq_array, nullptr));

    CUDA_CHECK(cudaMemcpyAsync(dev_q, query.data, q_num_elem * sizeof(dtype),
                               cudaMemcpyHostToDevice, data_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_k, key.data, k_num_elem * sizeof(dtype),
                               cudaMemcpyHostToDevice, data_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_v, value.data, v_num_elem * sizeof(dtype),
                               cudaMemcpyHostToDevice, data_stream));
    CUDA_CHECK(cudaStreamSynchronize(weight_stream));
    CUDA_CHECK(cudaStreamSynchronize(data_stream));

    CUDNN_CHECK(cudnnMultiHeadAttnForward(
        handle, attn_desc, -1, lo_win_idx, hi_win_idx, dev_q_seq_array,
        dev_k_seq_array, q_desc, dev_q, nullptr, k_desc, dev_k, v_desc, dev_v,
        o_desc, dev_o, size_weights, size_weights > 0 ? dev_w : nullptr,
        size_wkspace, dev_wkspace, 0, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyAsync(output.data, dev_o, o_num_elem * sizeof(dtype),
                               cudaMemcpyDeviceToHost, data_stream));

    CUDA_CHECK(cudaStreamDestroy(weight_stream));
    CUDNN_CHECK(cudnnDestroyAttnDescriptor(attn_desc));
    if (dropout > 0) {
        CUDNN_CHECK(cudnnDestroyDropoutDescriptor(drop_desc));
    }
    CUDNN_CHECK(cudnnDestroySeqDataDescriptor(q_desc));
    CUDNN_CHECK(cudnnDestroySeqDataDescriptor(k_desc));
    CUDNN_CHECK(cudnnDestroySeqDataDescriptor(v_desc));
    CUDNN_CHECK(cudnnDestroySeqDataDescriptor(o_desc));
    if (size_weights > 0) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(weight_desc));
    }
    CUDA_CHECK(cudaFree(dev_q));
    CUDA_CHECK(cudaFree(dev_k));
    CUDA_CHECK(cudaFree(dev_v));
    CUDA_CHECK(cudaFree(dev_o));
    if (dropout > 0) {
        CUDA_CHECK(cudaFree(dropout_buf));
    }
    if (size_weights > 0) {
        CUDA_CHECK(cudaFree(dev_w));
    }
    if (size_wkspace > 0) {
        CUDA_CHECK(cudaFree(dev_wkspace));
    }
    CUDA_CHECK(cudaFree(dev_q_seq_array));
    CUDA_CHECK(cudaFree(dev_k_seq_array));
    delete[] hi_win_idx;
    delete[] lo_win_idx;
    delete[] q_seq_array;
    delete[] k_seq_array;

    CUDA_CHECK(cudaStreamSynchronize(data_stream));
    CUDA_CHECK(cudaStreamDestroy(data_stream));

    return output;
}

template Tensor mha::standard<float>(MHA_PARAMS);
template Tensor mha::standard<double>(MHA_PARAMS);
