#include <ai3.hpp>
#include <algos.hpp>
#include <cuda_runtime.h>
#include <cuda_utils.cuh>
#include <cudnn.h>
#include <cudnn_utils.hpp>

const int WEIGHT_RANK = 3;

// shape
// 4 16 64
// stride
// 16 1 64
// finding index

// use stride and dim to properly copy
// host
//   shape: H * proj x embed dim
//   mem: H * proj x embed dim
// dev
//   shape: H x proj x embed dim
//   mem: proj x embed dim x H
void init_weights(cudnnHandle_t handle, cudnnAttnDescriptor_t attn_desc,
                  cudnnMultiHeadAttnWeightKind_t kind, uint count,
                  int size_weights, cudnnTensorDescriptor_t desc, void *dev_w,
                  void *host_data, uint num_heads, uint head_dim,
                  uint embed_dim, bool is_proj_weights) {
    int dim[WEIGHT_RANK], stride[WEIGHT_RANK];
    int ndim;
    float *weight_addr = nullptr;

    CUDNN_CHECK(cudnnGetMultiHeadAttnWeights(handle, attn_desc, kind,
                                             size_weights, dev_w, desc,
                                             (void **)&weight_addr));

    cudnnDataType_t data_type_unused;
    CUDNN_CHECK(cudnnGetTensorNdDescriptor(desc, WEIGHT_RANK, &data_type_unused,
                                           &ndim, dim, stride));

    assert(ndim == WEIGHT_RANK);

    if (is_proj_weights) {
        float *reordered_weights = new float[count / sizeof(float)];

        for (uint e = 0; e < embed_dim; ++e) {
            for (uint h = 0; h < num_heads; ++h) {
                for (uint d = 0; d < head_dim; ++d) {
                    reordered_weights[h * head_dim * embed_dim + d * embed_dim +
                                      e] =
                        ((float *)host_data)[e * num_heads * head_dim +
                                             h * head_dim + d];
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(weight_addr, reordered_weights, count,
                                   cudaMemcpyHostToDevice));
        delete[] reordered_weights;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(weight_addr, host_data, count,
                                   cudaMemcpyHostToDevice));
    }

    std::cout << "Shape: ";
    for (int i = 0; i < WEIGHT_RANK; ++i) {
        std::cout << dim[i] << " ";
    }
    std::cout << std::endl << "Stride: ";
    for (int i = 0; i < WEIGHT_RANK; ++i) {
        std::cout << stride[i] << " ";
    }
    std::cout << std::endl;
}

template <typename dtype>
Tensor
mha::standard(Tensor query, Tensor key, Tensor value, const Tensor &q_proj,
              const Tensor &k_proj, const Tensor &v_proj,
              const std::optional<const Tensor> &q_bias_in,
              const std::optional<const Tensor> &k_bias_in,
              const std::optional<const Tensor> &v_bias_in,
              const std::optional<const Tensor> &kbias,
              const std::optional<const Tensor> &vbias, const Tensor &out_proj,
              const std::optional<const Tensor> &out_bias,
              const bool add_zero_attn, const uint num_heads,
              const uint head_dim, std::optional<Tensor> &key_padding_mask,
              std::optional<Tensor> &attn_mask, const bool need_weights,
              const bool average_attn_weights, const bool is_causal,
              const bool need_to_project) { // TODO rename need_to_project to
                                            // need_to_project_input because we
                                            // always project output? myabe we
                                            // can't split it that fine though
    std::cout << "starting" << std::endl;
    // TODO use memcpyAsync after working initially
    uint batch_size, seq_len_q, embed_q, seq_len_k, embed_k, seq_len_v, embed_v;
    if (need_to_project) {
        errs::bail_if(query.shape.size() != RANK_NOT_PROJECTED,
                      "need to project in mha but rank of input is: ",
                      RANK_NOT_PROJECTED);
        batch_size = query.shape[0];
        seq_len_q = query.shape[1];
        seq_len_k = key.shape[1];
        seq_len_v = value.shape[1];
        embed_q = query.shape[2];
        embed_k = key.shape[2];
        embed_v = value.shape[2];
    } else {
        errs::bail("not handling projected yet");
        errs::bail_if(
            query.shape.size() != RANK_PROJECTED,
            "input is projected but rank of input is: ", RANK_PROJECTED);
    }
    uint proj_q = embed_q / num_heads;
    uint proj_k = embed_k / num_heads;
    uint proj_v = embed_v / num_heads;
    uint proj_o = embed_v;
    uint embed_o = embed_v;
    float sm_scaler = 1.0f / std::sqrt(proj_v);

    const bool proj_bias = q_bias_in.has_value() && k_bias_in.has_value() &&
                           v_bias_in.has_value() && out_bias.has_value();

    Tensor output({batch_size, seq_len_q, embed_o}, query.scalar_type);
    std::fill((float *)output.data, (float *)output.data + output.count(), 0);

    cudnnHandle_t handle = (cudnnHandle_t)Context::cudnn_handle_t();
    cudnnAttnDescriptor_t attn_desc;
    cudnnDropoutDescriptor_t drop_desc;
    cudnnSeqDataDescriptor_t q_desc;
    cudnnSeqDataDescriptor_t k_desc;
    cudnnSeqDataDescriptor_t v_desc;
    cudnnSeqDataDescriptor_t o_desc;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnDataType_t comp_prec = CUDNN_DATA_FLOAT;

    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnCreateAttnDescriptor(&attn_desc));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&drop_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&q_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&k_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&v_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&o_desc));
    uint *qSeqArray = nullptr;
    uint *kSeqArray = nullptr;

    uint *loWinIdx = nullptr;
    uint *hiWinIdx = nullptr;
    unsigned attn_mode;
    if (proj_bias && need_to_project) {
        attn_mode =
            CUDNN_ATTN_ENABLE_PROJ_BIASES | CUDNN_ATTN_QUERYMAP_ALL_TO_ONE;
    } else {
        attn_mode =
            CUDNN_ATTN_DISABLE_PROJ_BIASES | CUDNN_ATTN_QUERYMAP_ALL_TO_ONE;
    }

    size_t qo_tokens = size_t(seq_len_q) * batch_size;
    size_t kv_tokens = size_t(seq_len_k) * batch_size;

    size_t q_num_elem = qo_tokens * embed_q;
    size_t k_num_elem = kv_tokens * embed_k;
    size_t v_num_elem = kv_tokens * embed_v;
    size_t o_num_elem = qo_tokens * embed_o;

    size_t q_num_weights = embed_q * proj_q * num_heads;
    size_t k_num_weights = embed_k * proj_k * num_heads;
    size_t v_num_weights = embed_v * proj_v * num_heads;
    size_t o_num_weights = embed_o * proj_o;

    size_t q_bias_len = embed_q;
    size_t k_bias_len = embed_k;
    size_t v_bias_len = embed_v;
    size_t o_bias_len = embed_o;

    // TODO make this dtype
    float *dev_q = nullptr;
    float *dev_k = nullptr;
    float *dev_v = nullptr;
    float *dev_o = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&dev_q, q_num_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_k, k_num_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_v, v_num_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_o, o_num_elem * sizeof(float)));

    size_t dropout_buf_size;
    void *dropout_buf = nullptr;
    float dropout_rate = 0;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &dropout_buf_size));
    CUDA_CHECK(cudaMalloc(&dropout_buf, dropout_buf_size));
    CUDNN_CHECK(cudnnSetDropoutDescriptor(drop_desc, handle, dropout_rate,
                                          dropout_buf, dropout_buf_size, 0));
    CUDNN_CHECK(cudnnSetAttnDescriptor(
        attn_desc, attn_mode, num_heads, sm_scaler, data_type, comp_prec,
        CUDNN_DEFAULT_MATH, dropout_rate > 0 ? drop_desc : nullptr, nullptr,
        embed_q, embed_k, embed_v, proj_q, proj_k, proj_v, proj_o, seq_len_q,
        seq_len_k, batch_size, 1));

    size_t size_weights = 0, size_wkspace = 0;
    float *dev_w = nullptr;
    float *dev_wkspace = nullptr;
    CUDNN_CHECK(cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &size_weights,
                                             &size_wkspace, nullptr));
    if (size_weights > 0) {
        CUDA_CHECK(cudaMalloc((void **)&dev_w, size_weights));
    }
    if (size_wkspace > 0) {
        CUDA_CHECK(cudaMalloc((void **)&dev_wkspace, size_wkspace));
    }

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    init_weights(handle, attn_desc, CUDNN_MH_ATTN_Q_WEIGHTS,
                 q_num_weights * sizeof(float), size_weights, desc, dev_w,
                 q_proj.data, num_heads, head_dim, embed_q, true);
    init_weights(handle, attn_desc, CUDNN_MH_ATTN_K_WEIGHTS,
                 k_num_weights * sizeof(float), size_weights, desc, dev_w,
                 k_proj.data, num_heads, head_dim, embed_k, true);
    init_weights(handle, attn_desc, CUDNN_MH_ATTN_V_WEIGHTS,
                 v_num_weights * sizeof(float), size_weights, desc, dev_w,
                 v_proj.data, num_heads, head_dim, embed_v, true);
    init_weights(handle, attn_desc, CUDNN_MH_ATTN_O_WEIGHTS,
                 o_num_weights * sizeof(float), size_weights, desc, dev_w,
                 out_proj.data, num_heads, head_dim, embed_o, true);
    if (proj_bias) {
        init_weights(handle, attn_desc, CUDNN_MH_ATTN_Q_BIASES,
                     q_bias_len * sizeof(float), size_weights, desc, dev_w,
                     q_bias_in->data, 0, 0, 0, false);
        init_weights(handle, attn_desc, CUDNN_MH_ATTN_K_BIASES,
                     k_bias_len * sizeof(float), size_weights, desc, dev_w,
                     k_bias_in->data, 0, 0, 0, false);
        init_weights(handle, attn_desc, CUDNN_MH_ATTN_V_BIASES,
                     v_bias_len * sizeof(float), size_weights, desc, dev_w,
                     v_bias_in->data, 0, 0, 0, false);
        init_weights(handle, attn_desc, CUDNN_MH_ATTN_O_BIASES,
                     o_bias_len * sizeof(float), size_weights, desc, dev_w,
                     out_bias->data, 0, 0, 0, false);
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
    CUDA_CHECK(cudaMemcpy(dev_q_seq_array, q_seq_array,
                          batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&dev_k_seq_array, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev_k_seq_array, k_seq_array,
                          batch_size * sizeof(int), cudaMemcpyHostToDevice));

    int dim_a[CUDNN_SEQDATA_DIM_COUNT];
    cudnnSeqDataAxis_t data_axes[CUDNN_SEQDATA_DIM_COUNT];
    data_axes[0] = CUDNN_SEQDATA_BEAM_DIM;
    data_axes[1] = CUDNN_SEQDATA_BATCH_DIM;
    data_axes[2] = CUDNN_SEQDATA_TIME_DIM;
    data_axes[3] = CUDNN_SEQDATA_VECT_DIM;
    dim_a[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dim_a[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
    dim_a[CUDNN_SEQDATA_TIME_DIM] = seq_len_q;
    dim_a[CUDNN_SEQDATA_VECT_DIM] = embed_q;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        q_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a, data_axes,
        batch_size, q_seq_array, NULL));

    dim_a[CUDNN_SEQDATA_VECT_DIM] = embed_o;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        o_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a, data_axes,
        batch_size, q_seq_array, NULL));

    dim_a[CUDNN_SEQDATA_TIME_DIM] = seq_len_k;
    dim_a[CUDNN_SEQDATA_VECT_DIM] = embed_k;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        k_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a, data_axes,
        batch_size, k_seq_array, NULL));

    dim_a[CUDNN_SEQDATA_TIME_DIM] = seq_len_k;
    dim_a[CUDNN_SEQDATA_VECT_DIM] = embed_v;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        v_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a, data_axes,
        batch_size, k_seq_array, NULL));

    CUDA_CHECK(cudaMemcpy(dev_q, query.data, sizeof(float) * q_num_elem,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_k, key.data, sizeof(float) * k_num_elem,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_v, value.data, sizeof(float) * v_num_elem,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDNN_CHECK(cudnnMultiHeadAttnForward(
        handle, attn_desc, -1, lo_win_idx, hi_win_idx, dev_q_seq_array,
        dev_k_seq_array, q_desc, dev_q, nullptr, k_desc, dev_k, v_desc, dev_v,
        o_desc, dev_o, size_weights, size_weights > 0 ? dev_w : nullptr,
        size_wkspace, dev_wkspace, 0, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data, dev_o, o_num_elem * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float *odata = (float *)output.data;
    // for (int i = 0; i < output.count(); i++) {
    //     std::cout << odata[i] << std::endl;
    // }
    // TODO destroy things
    return output;
}

template Tensor mha::standard<float>(MHA_PARAMS);
template Tensor mha::standard<double>(MHA_PARAMS);
