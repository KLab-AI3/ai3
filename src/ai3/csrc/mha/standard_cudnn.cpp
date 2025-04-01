#include <ai3.hpp>
#include <algos.hpp>
#include <cuda_runtime.h>
#include <cuda_utils.hpp>
#include <cudnn.h>
#include <cudnn_utils.hpp>
#include <numeric>
#include <optional>

const int WEIGHT_RANK = 3;
const int NUM_PROJECTION_WEIGHTS = 4;

template <typename dtype>
dtype *dev_dw_to_host(cudnnHandle_t handle, cudnnAttnDescriptor_t attn_desc,
                      cudnnMultiHeadAttnWeightKind_t kind, bool is_proj_weights,
                      size_t size_all_weights, cudnnTensorDescriptor_t desc,
                      void *dev_dw, void *host_data, uint num_heads,
                      uint head_dim, uint embed_dim, cudaStream_t stream) {
    std::cout << "enter" << std::endl;
    int dim[WEIGHT_RANK], stride[WEIGHT_RANK];
    int ndim;
    dtype *weight_addr = nullptr;
    CUDNN_CHECK(cudnnGetMultiHeadAttnWeights(handle, attn_desc, kind,
                                             size_all_weights, dev_dw, desc,
                                             (void **)&weight_addr));

    cudnnDataType_t data_type_unused;
    CUDNN_CHECK(cudnnGetTensorNdDescriptor(desc, WEIGHT_RANK, &data_type_unused,
                                           &ndim, dim, stride));
    // TODO can do a check here of stride to see if it is actually different
    // if not can just copy
    uint num_weights = num_heads * head_dim * embed_dim;
    assert(ndim == WEIGHT_RANK);
    dtype *buffer = nullptr;

    if (is_proj_weights) {
        std::cout << "weights" << std::endl;
        CUDA_CHECK(cudaMalloc((void **)&buffer, num_weights * sizeof(dtype)));
        CUDA_CHECK(
            cudaMemset((void **)&buffer, 0, num_weights * sizeof(dtype)));
        std::cout << "malloced" << std::endl;
        transpose_call(buffer, weight_addr, embed_dim, head_dim * num_heads,
                       stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "kernel" << std::endl;
        CUDA_CHECK(cudaMemcpyAsync(host_data, buffer,
                                   num_weights * sizeof(dtype),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "cpy" << std::endl;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(host_data, weight_addr,
                                   num_weights * sizeof(dtype),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "cpy bias" << std::endl;
    }
    std::cout << "leave" << std::endl;
    return buffer;
}

template <typename dtype>
dtype *host_w_to_dev(cudnnHandle_t handle, cudnnAttnDescriptor_t attn_desc,
                     cudnnMultiHeadAttnWeightKind_t kind, bool is_proj_weights,
                     size_t size_all_weights, cudnnTensorDescriptor_t desc,
                     void *dev_w, void *host_data, uint num_heads,
                     uint head_dim, uint embed_dim, bool identity,
                     cudaStream_t stream) {
    int dim[WEIGHT_RANK], stride[WEIGHT_RANK];
    int ndim;
    dtype *weight_addr = nullptr;
    CUDNN_CHECK(cudnnGetMultiHeadAttnWeights(handle, attn_desc, kind,
                                             size_all_weights, dev_w, desc,
                                             (void **)&weight_addr));

    cudnnDataType_t data_type_unused;
    CUDNN_CHECK(cudnnGetTensorNdDescriptor(desc, WEIGHT_RANK, &data_type_unused,
                                           &ndim, dim, stride));

    uint num_weights = num_heads * head_dim * embed_dim;
    assert(ndim == WEIGHT_RANK);
    dtype *buffer = nullptr;
    if (is_proj_weights) {
        if (identity) {
            fill_identity_call(weight_addr, embed_dim, num_heads * head_dim,
                               stream);
        } else {
            CUDA_CHECK(
                cudaMalloc((void **)&buffer, num_weights * sizeof(dtype)));
            CUDA_CHECK(cudaMemcpyAsync(buffer, host_data,
                                       num_weights * sizeof(dtype),
                                       cudaMemcpyHostToDevice, stream));
            transpose_call(weight_addr, buffer, head_dim * num_heads, embed_dim,
                           stream);
        }
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
    return buffer;
}

// TODO need to check with the need_to_project_input in order to see if some
// gradients are needed or not, can skip over weights if not needed?
// not sure how that would work though because we need the torch operations
// in the forward that we don't handle to be accounted for
template <typename dtype>
std::array<std::optional<Tensor>, mha::NUM_GRAD>
operate(Tensor query, Tensor key, Tensor value,
        const mha::MemFormat input_format, const Tensor &q_proj,
        const Tensor &k_proj, const Tensor &v_proj,
        const std::optional<const Tensor> &q_bias_in,
        const std::optional<const Tensor> &k_bias_in,
        const std::optional<const Tensor> &v_bias_in,
        const std::optional<const Tensor> &k_bias,
        const std::optional<const Tensor> &v_bias, const Tensor &o_proj,
        const std::optional<const Tensor> &o_bias, const bool add_zero_attn,
        const uint num_heads, const float dropout,
        const std::optional<const Tensor> &key_padding_mask,
        const std::optional<const Tensor> &attn_mask, const bool need_weights,
        const bool average_attn_weights, const bool is_causal,
        const bool need_to_project_input, const bool is_training,
        const intptr_t do_address) {
    ensure_same_type(query, key, value);
    errs::bail_if(need_weights, "no support for attention weights");
    errs::bail_if(need_weights and average_attn_weights,
                  "no support for average attention weights");
    errs::bail_if(key_padding_mask.has_value(),
                  "no support for key padding mask");
    errs::bail_if(!is_causal && attn_mask.has_value(),
                  "no support for attention mask");

    std::cout << "begin operate" << std::endl;
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
    const bool use_dropout = dropout > 0 && is_training;

    const bool proj_bias = q_bias_in.has_value() && k_bias_in.has_value() &&
                           v_bias_in.has_value() && o_bias.has_value();

    std::vector<uint> o_shape(sample_dims::MHA_NOT_PROJECTED);
    if (input_format == mha::MemFormat::NSE ||
        input_format == mha::MemFormat::NSHD) {
        o_shape = {batch_size, seq_len_q, embed_o};
    } else if (input_format == mha::MemFormat::SNE ||
               input_format == mha::MemFormat::SNHD) {
        o_shape = {seq_len_q, batch_size, embed_o};
    }

    cudnnHandle_t handle = (cudnnHandle_t)Context::cudnn_handle_t();
    StreamSwapper ss = StreamSwapper();
    cudnnAttnDescriptor_t attn_desc;
    cudnnDropoutDescriptor_t drop_desc;
    cudnnSeqDataDescriptor_t q_desc;
    cudnnSeqDataDescriptor_t k_desc;
    cudnnSeqDataDescriptor_t v_desc;
    cudnnSeqDataDescriptor_t o_desc;
    cudnnDataType_t data_type = cudnn_data_type<dtype>();
    cudnnDataType_t comp_prec = cudnn_data_type<dtype>();

    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnCreateAttnDescriptor(&attn_desc));
    if (use_dropout) {
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&drop_desc));
    }
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&q_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&k_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&v_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&o_desc));

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

    dtype *dev_q = nullptr, *dev_k = nullptr, *dev_v = nullptr,
          *dev_o = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&dev_q, q_num_elem * sizeof(dtype)));
    CUDA_CHECK(cudaMalloc((void **)&dev_k, k_num_elem * sizeof(dtype)));
    CUDA_CHECK(cudaMalloc((void **)&dev_v, v_num_elem * sizeof(dtype)));
    CUDA_CHECK(cudaMalloc((void **)&dev_o, o_num_elem * sizeof(dtype)));

    size_t dropout_buf_size;
    void *dropout_buf = nullptr;
    if (use_dropout) {
        CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &dropout_buf_size));
        CUDA_CHECK(cudaMalloc(&dropout_buf, dropout_buf_size));
        CUDNN_CHECK(cudnnSetDropoutDescriptor(
            drop_desc, handle, dropout, dropout_buf, dropout_buf_size, 0));
    }
    CUDNN_CHECK(cudnnSetAttnDescriptor(
        attn_desc, attn_mode, num_heads, sm_scaler, data_type, comp_prec,
        CUDNN_DEFAULT_MATH, use_dropout ? drop_desc : nullptr, nullptr, embed_q,
        embed_k, embed_v, proj_q, proj_k, proj_v, proj_o, seq_len_q, seq_len_k,
        batch_size, 1));

    size_t size_weights = 0, size_wkspace = 0, size_reserve = 0;
    dtype *dev_w = nullptr, *dev_wkspace = nullptr, *dev_reserve = nullptr;
    CUDNN_CHECK(cudnnGetMultiHeadAttnBuffers(
        handle, attn_desc, &size_weights, &size_wkspace,
        is_training ? &size_reserve : nullptr));

    uint total_num_weights =
        o_num_weights + q_num_weights + k_num_weights + v_num_weights;
    if (proj_bias) {
        total_num_weights += q_bias_len + k_bias_len + v_bias_len + o_bias_len;
    }
    assert(size_weights / sizeof(dtype) == total_num_weights);

    cudnnTensorDescriptor_t weight_desc;
    dtype *buffers[NUM_PROJECTION_WEIGHTS] = {nullptr};
    if (size_weights > 0) {
        CUDA_CHECK(cudaMalloc((void **)&dev_w, size_weights));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&weight_desc));
        buffers[0] = host_w_to_dev<dtype>(
            handle, attn_desc, CUDNN_MH_ATTN_Q_WEIGHTS, true, size_weights,
            weight_desc, dev_w, q_proj.data, num_heads, proj_q, embed_q,
            !need_to_project_input, ss());
        buffers[1] = host_w_to_dev<dtype>(
            handle, attn_desc, CUDNN_MH_ATTN_K_WEIGHTS, true, size_weights,
            weight_desc, dev_w, k_proj.data, num_heads, proj_k, embed_k,
            !need_to_project_input, ss());
        buffers[2] = host_w_to_dev<dtype>(
            handle, attn_desc, CUDNN_MH_ATTN_V_WEIGHTS, true, size_weights,
            weight_desc, dev_w, v_proj.data, num_heads, proj_v, embed_v,
            !need_to_project_input, ss());
        buffers[3] = host_w_to_dev<dtype>(
            handle, attn_desc, CUDNN_MH_ATTN_O_WEIGHTS, true, size_weights,
            weight_desc, dev_w, o_proj.data, 1, proj_o, embed_o, false, ss());
        if (proj_bias) {
            host_w_to_dev<dtype>(handle, attn_desc, CUDNN_MH_ATTN_Q_BIASES,
                                 false, size_weights, weight_desc, dev_w,
                                 q_bias_in->data, q_bias_len, 1, 1,
                                 !need_to_project_input, ss());
            host_w_to_dev<dtype>(handle, attn_desc, CUDNN_MH_ATTN_K_BIASES,
                                 false, size_weights, weight_desc, dev_w,
                                 k_bias_in->data, k_bias_len, 1, 1,
                                 !need_to_project_input, ss());
            host_w_to_dev<dtype>(handle, attn_desc, CUDNN_MH_ATTN_V_BIASES,
                                 false, size_weights, weight_desc, dev_w,
                                 v_bias_in->data, k_bias_len, 1, 1,
                                 !need_to_project_input, ss());
            host_w_to_dev<dtype>(handle, attn_desc, CUDNN_MH_ATTN_O_BIASES,
                                 false, size_weights, weight_desc, dev_w,
                                 o_bias->data, o_bias_len, 1, 1, false, ss());
        }
    }
    if (size_wkspace > 0) {
        CUDA_CHECK(cudaMalloc((void **)&dev_wkspace, size_wkspace));
    }
    if (size_reserve > 0) {
        CUDA_CHECK(cudaMalloc((void **)&dev_reserve, size_reserve));
    }

    int *q_seq_array = new int[batch_size]();
    std::fill(q_seq_array, q_seq_array + batch_size, seq_len_q);
    int *k_seq_array = new int[batch_size]();
    std::fill(k_seq_array, k_seq_array + batch_size, seq_len_k);

    int *lo_win_idx = new int[seq_len_q]();
    int *hi_win_idx = new int[seq_len_q];
    if (is_causal) {
        std::iota(hi_win_idx, hi_win_idx + seq_len_q, 1);
    } else {
        std::fill(hi_win_idx, hi_win_idx + seq_len_q, seq_len_k);
    }

    int *dev_q_seq_array = nullptr;
    int *dev_k_seq_array = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&dev_q_seq_array, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(dev_q_seq_array, q_seq_array,
                               batch_size * sizeof(int), cudaMemcpyHostToDevice,
                               ss()));

    CUDA_CHECK(cudaMalloc((void **)&dev_k_seq_array, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(dev_k_seq_array, k_seq_array,
                               batch_size * sizeof(int), cudaMemcpyHostToDevice,
                               ss()));

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
                               cudaMemcpyHostToDevice, ss()));
    CUDA_CHECK(cudaMemcpyAsync(dev_k, key.data, k_num_elem * sizeof(dtype),
                               cudaMemcpyHostToDevice, ss()));
    CUDA_CHECK(cudaMemcpyAsync(dev_v, value.data, v_num_elem * sizeof(dtype),
                               cudaMemcpyHostToDevice, ss()));

    ss.sync();
    for (int i = 0; i < NUM_PROJECTION_WEIGHTS; i++) {
        CUDA_CHECK(cudaFree(buffers[i]));
    }
    CUDNN_CHECK(cudnnMultiHeadAttnForward(
        handle, attn_desc, -1,                            //
        lo_win_idx, hi_win_idx,                           //
        dev_q_seq_array, dev_k_seq_array,                 //
        q_desc, dev_q,                                    //
        nullptr,                                          //
        k_desc, dev_k,                                    //
        v_desc, dev_v,                                    //
        o_desc, dev_o,                                    //
        size_weights, size_weights > 0 ? dev_w : nullptr, //
        size_wkspace, dev_wkspace, size_reserve,          //
        size_reserve > 0 ? dev_reserve : nullptr));

    std::array<std::optional<Tensor>, mha::NUM_GRAD> out{std::nullopt};
    if (!is_training) {
        CUDA_CHECK(cudaDeviceSynchronize());
        Tensor output(std::move(o_shape), query.scalar_type);
        CUDA_CHECK(cudaMemcpyAsync(output.data, dev_o,
                                   o_num_elem * sizeof(dtype),
                                   cudaMemcpyDeviceToHost, ss()));
        out[0] = std::optional<Tensor>(std::move(output));
    } else {
        dtype *dev_dq = nullptr, *dev_dk = nullptr, *dev_dv = nullptr,
              *dev_do = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&dev_dq, q_num_elem * sizeof(dtype)));
        CUDA_CHECK(cudaMalloc((void **)&dev_dk, k_num_elem * sizeof(dtype)));
        CUDA_CHECK(cudaMalloc((void **)&dev_dv, v_num_elem * sizeof(dtype)));
        CUDA_CHECK(cudaMalloc((void **)&dev_do, o_num_elem * sizeof(dtype)));
        CUDA_CHECK(cudaMemcpy(dev_do, reinterpret_cast<void *>(do_address),
                              o_num_elem * sizeof(dtype),
                              cudaMemcpyHostToDevice));

        dtype *dev_dw = nullptr;
        if (size_weights > 0) {
            CUDA_CHECK(cudaMalloc((void **)&dev_dw, size_weights));
        }
        CUDNN_CHECK(cudnnMultiHeadAttnBackwardData(
            handle, attn_desc,                                      //
            lo_win_idx, hi_win_idx,                                 //
            dev_q_seq_array, dev_k_seq_array,                       //
            o_desc, dev_do,                                         //
            q_desc, dev_dq, dev_q,                                  //
            k_desc, dev_dk, dev_k,                                  //
            v_desc, dev_dv, dev_v,                                  //
            size_weights, size_weights > 0 ? dev_w : nullptr,       //
            size_wkspace, size_wkspace > 0 ? dev_wkspace : nullptr, //
            size_reserve, size_reserve > 0 ? dev_reserve : nullptr));
        CUDNN_CHECK(cudnnMultiHeadAttnBackwardWeights(
            handle, attn_desc, CUDNN_WGRAD_MODE_SET,                //
            q_desc, dev_q,                                          //
            k_desc, dev_k,                                          //
            v_desc, dev_v,                                          //
            o_desc, dev_do,                                         //
            size_weights, size_weights > 0 ? dev_w : nullptr,       //
            size_weights > 0 ? dev_dw : nullptr,                    //
            size_wkspace, size_wkspace > 0 ? dev_wkspace : nullptr, //
            size_reserve, size_reserve > 0 ? dev_reserve : nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "copying" << std::endl;
        Tensor dq(std::move(query.shape), query.scalar_type);
        Tensor dk(std::move(key.shape), key.scalar_type);
        Tensor dv(std::move(value.shape), value.scalar_type);
        CUDA_CHECK(cudaMemcpyAsync(dq.data, dev_dq, q_num_elem * sizeof(dtype),
                                   cudaMemcpyDeviceToHost, ss()));
        CUDA_CHECK(cudaMemcpyAsync(dk.data, dev_dk, k_num_elem * sizeof(dtype),
                                   cudaMemcpyDeviceToHost, ss()));
        CUDA_CHECK(cudaMemcpyAsync(dv.data, dev_dv, v_num_elem * sizeof(dtype),
                                   cudaMemcpyDeviceToHost, ss()));
        out[0] = std::optional<Tensor>(std::move(dq));
        out[1] = std::optional<Tensor>(std::move(dk));
        out[2] = std::optional<Tensor>(std::move(dv));

        dtype *buffers[NUM_PROJECTION_WEIGHTS] = {nullptr};
        if (size_weights > 0) {
            Tensor dq_proj(std::move(q_proj.shape), q_proj.scalar_type);
            Tensor dk_proj(std::move(k_proj.shape), k_proj.scalar_type);
            Tensor dv_proj(std::move(v_proj.shape), v_proj.scalar_type);
            Tensor do_proj(std::move(o_proj.shape), o_proj.scalar_type);
            std::cout << "dev_dw_to_host calls" << std::endl;
            buffers[0] = dev_dw_to_host<dtype>(
                handle, attn_desc, CUDNN_MH_ATTN_Q_WEIGHTS, true, size_weights,
                weight_desc, dev_dw, dq_proj.data, num_heads, proj_q, embed_q,
                ss());
            buffers[1] = dev_dw_to_host<dtype>(
                handle, attn_desc, CUDNN_MH_ATTN_K_WEIGHTS, true, size_weights,
                weight_desc, dev_dw, dk_proj.data, num_heads, proj_k, embed_k,
                ss());
            buffers[2] = dev_dw_to_host<dtype>(
                handle, attn_desc, CUDNN_MH_ATTN_V_WEIGHTS, true, size_weights,
                weight_desc, dev_dw, dv_proj.data, num_heads, proj_v, embed_v,
                ss());
            buffers[3] = dev_dw_to_host<dtype>(
                handle, attn_desc, CUDNN_MH_ATTN_O_WEIGHTS, true, size_weights,
                weight_desc, dev_dw, do_proj.data, 1, proj_o, embed_o, ss());
            out[3] = std::optional<Tensor>(std::move(dq_proj));
            out[4] = std::optional<Tensor>(std::move(dk_proj));
            out[5] = std::optional<Tensor>(std::move(dv_proj));
            out[6] = std::optional<Tensor>(std::move(do_proj));

            if (proj_bias) {
                Tensor dq_bias(std::move(q_bias_in->shape),
                               q_bias_in->scalar_type);
                Tensor dk_bias(std::move(k_bias_in->shape),
                               k_bias_in->scalar_type);
                Tensor dv_bias(std::move(v_bias_in->shape),
                               v_bias_in->scalar_type);
                Tensor do_bias(std::move(o_bias->shape), o_bias->scalar_type);
                dev_dw_to_host<dtype>(handle, attn_desc, CUDNN_MH_ATTN_Q_BIASES,
                                      false, size_weights, weight_desc, dev_dw,
                                      dq_bias.data, 1, 1, q_bias_len, ss());
                dev_dw_to_host<dtype>(handle, attn_desc, CUDNN_MH_ATTN_K_BIASES,
                                      false, size_weights, weight_desc, dev_dw,
                                      dk_bias.data, 1, 1, k_bias_len, ss());
                dev_dw_to_host<dtype>(handle, attn_desc, CUDNN_MH_ATTN_V_BIASES,
                                      false, size_weights, weight_desc, dev_dw,
                                      dv_bias.data, 1, 1, v_bias_len, ss());
                dev_dw_to_host<dtype>(handle, attn_desc, CUDNN_MH_ATTN_O_BIASES,
                                      false, size_weights, weight_desc, dev_dw,
                                      do_bias.data, 1, 1, o_bias_len, ss());
                out[7] = std::optional<Tensor>(std::move(dq_bias));
                out[8] = std::optional<Tensor>(std::move(dk_bias));
                out[9] = std::optional<Tensor>(std::move(dv_bias));
                out[10] = std::optional<Tensor>(std::move(do_bias));
                std::cout << "dev_dw_to_host done bias" << std::endl;
            }
        }
        ss.sync();
        for (int i = 0; i < NUM_PROJECTION_WEIGHTS; i++) {
            CUDA_CHECK(cudaFree(buffers[i]));
        }
        CUDA_CHECK(cudaFree(dev_dq));
        CUDA_CHECK(cudaFree(dev_dk));
        CUDA_CHECK(cudaFree(dev_dv));
        CUDA_CHECK(cudaFree(dev_do));
        CUDA_CHECK(cudaFree(dev_dw));
    }

    CUDNN_CHECK(cudnnDestroyAttnDescriptor(attn_desc));
    if (use_dropout) {
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
    if (use_dropout) {
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
    ss.sync();
    CUDA_CHECK(cudaFree(dev_o));

    delete[] hi_win_idx;
    delete[] lo_win_idx;
    delete[] q_seq_array;
    delete[] k_seq_array;

    std::cout << "done operate" << std::endl;
    return out;
}

template <typename dtype>
Tensor mha::standard(Tensor query, Tensor key, Tensor value,
                     const mha::MemFormat input_format, const Tensor &q_proj,
                     const Tensor &k_proj, const Tensor &v_proj,
                     const std::optional<const Tensor> &q_bias_in,
                     const std::optional<const Tensor> &k_bias_in,
                     const std::optional<const Tensor> &v_bias_in,
                     const std::optional<const Tensor> &k_bias,
                     const std::optional<const Tensor> &v_bias,
                     const Tensor &out_proj,
                     const std::optional<const Tensor> &out_bias,
                     const bool add_zero_attn, const uint num_heads,
                     const float dropout,
                     const std::optional<const Tensor> &key_padding_mask,
                     const std::optional<const Tensor> &attn_mask,
                     const bool need_weights, const bool average_attn_weights,
                     const bool is_causal, const bool need_to_project_input) {
    return std::move(*operate<dtype>(
        std::move(query), std::move(key), std::move(value), input_format,
        q_proj, k_proj, v_proj, q_bias_in, k_bias_in, v_bias_in, k_bias, v_bias,
        out_proj, out_bias, add_zero_attn, num_heads, dropout, key_padding_mask,
        attn_mask, need_weights, average_attn_weights, is_causal,
        need_to_project_input, false, 0)[0]);
}

template <typename dtype>
std::array<std::optional<Tensor>, mha::NUM_GRAD> mha::standard_backward(
    const intptr_t do_address, Tensor query, Tensor key, Tensor value,
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
    std::cout << "standard_backward" << std::endl;
    return operate<dtype>(
        std::move(query), std::move(key), std::move(value), input_format,
        q_proj, k_proj, v_proj, q_bias_in, k_bias_in, v_bias_in, k_bias, v_bias,
        out_proj, out_bias, add_zero_attn, num_heads, dropout, key_padding_mask,
        attn_mask, need_weights, average_attn_weights, is_causal,
        need_to_project_input, true, do_address);
}

template Tensor mha::standard<float>(MHA_PARAMS);
template Tensor mha::standard<double>(MHA_PARAMS);
template std::array<std::optional<Tensor>, mha::NUM_GRAD>
mha::standard_backward<float>(const intptr_t, MHA_PARAMS);
template std::array<std::optional<Tensor>, mha::NUM_GRAD>
mha::standard_backward<double>(const intptr_t, MHA_PARAMS);
