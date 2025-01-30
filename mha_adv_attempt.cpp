#include <cmath>
#include <cstdlib>
#include <cudnn.h>
#include <iostream>

#define CUDA_CHECK(status)                                                     \
    {                                                                          \
        if (status != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << ": " << cudaGetErrorString(status) << std::endl;      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    }

#define CUDNN_CHECK(status)                                                    \
    {                                                                          \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudnnGetErrorString(status) << std::endl;     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    }
const int WEIGHT_RANK = 3;
const int BIAS_RANK = 3;

// weight
// shape 4 16 64
// stride 16 1 64 meaning CHW shape with HWC in memory
// bias (same as array)
// shape 1 64 1
// stride 64 1 1
template <int rank>
void init_weights(cudnnHandle_t handle, cudnnAttnDescriptor_t attn_desc,
                  cudnnMultiHeadAttnWeightKind_t kind, int count,
                  int size_weights, cudnnTensorDescriptor_t desc, void *dev_w,
                  void *host_data) {
    int dim[rank], stride[rank];
    int ndim;
    float *weight_addr = nullptr;

    CUDNN_CHECK(cudnnGetMultiHeadAttnWeights(handle, attn_desc, kind,
                                             size_weights, dev_w, desc,
                                             (void **)&weight_addr));
    cudnnDataType_t data_type_unused = CUDNN_DATA_FLOAT;
    CUDNN_CHECK(cudnnGetTensorNdDescriptor(desc, rank, &data_type_unused, &ndim,
                                           dim, stride));
    CUDA_CHECK(
        cudaMemcpy(weight_addr, host_data, count, cudaMemcpyHostToDevice));
}

int main() {
    // proj bias and proj weights implemented
    // do in ai3!!
    // TODO make this in a input shape vector {batch_size, seq_len, embed}
    int batch_size = 2;
    int num_heads = 4;
    int seq_len_q = 10;
    int seq_len_k = 10;
    int embed_q = 64; // qSize
    int embed_k = 64; // kSize
    int embed_v = 64; // vSize
    int proj_q = embed_q / num_heads;
    int proj_k = embed_k / num_heads;
    int proj_v = embed_v / num_heads;
    int proj_o = embed_v;
    int embed_o = proj_o;
    bool is_train = false;
    bool proj_bias = true;

    double sm_scaler = 1.0f / std::sqrt(proj_v);
    float dropout_rate = 0;

    cudnnHandle_t handle;
    cudnnAttnDescriptor_t attn_desc;
    cudnnDropoutDescriptor_t drop_desc; // right now kind of pointless just
                                        // focusing on inference for now
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

    size_t dropoutBufSize;
    void *dropoutBuf;

    int *qSeqArray = nullptr;
    int *kSeqArray = nullptr;

    int *loWinIdx = nullptr;
    int *hiWinIdx = nullptr;
    unsigned attn_mode;
    if (proj_bias) {
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
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &dropout_buf_size));
    CUDA_CHECK(cudaMalloc(&dropout_buf, dropout_buf_size));
    CUDNN_CHECK(cudnnSetDropoutDescriptor(drop_desc, handle, dropout_rate,
                                          dropout_buf, dropout_buf_size, 0));

    CUDNN_CHECK(cudnnSetAttnDescriptor(
        attn_desc, attn_mode, num_heads, sm_scaler, data_type, comp_prec,
        CUDNN_DEFAULT_MATH, is_train && dropout_rate > 0 ? drop_desc : nullptr,
        nullptr, embed_q, embed_k, embed_v, proj_q, proj_k, proj_v, proj_o,
        seq_len_q, seq_len_k, batch_size, 1));

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
    const int w_rank = 3;
    int dim_w[w_rank], stride_w[w_rank];
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    cudnnDataType_t data_type_unused;
    float *weight_addr = nullptr;
    const int num_types_of_weights = 4;
    cudnnMultiHeadAttnWeightKind_t w_kind[num_types_of_weights] = {
        CUDNN_MH_ATTN_Q_WEIGHTS, CUDNN_MH_ATTN_K_WEIGHTS,
        CUDNN_MH_ATTN_V_WEIGHTS, CUDNN_MH_ATTN_O_WEIGHTS};
    float *q_w_data = new float[q_num_weights];
    float *q_b_data = new float[q_bias_len];
    for (size_t i = 0; i < q_num_weights; i++) {
        q_w_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < q_bias_len; i++) {
        q_b_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    init_weights<WEIGHT_RANK>(handle, attn_desc, CUDNN_MH_ATTN_Q_WEIGHTS,
                              q_num_weights * sizeof(float), size_weights, desc,
                              dev_w, q_w_data);
    init_weights<BIAS_RANK>(handle, attn_desc, CUDNN_MH_ATTN_Q_BIASES,
                            q_bias_len * sizeof(float), size_weights, desc,
                            dev_w, q_b_data);

    float *k_w_data = new float[k_num_weights];
    float *k_b_data = new float[k_bias_len];
    for (size_t i = 0; i < k_num_weights; i++) {
        k_w_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < k_bias_len; i++) {
        k_b_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    init_weights<WEIGHT_RANK>(handle, attn_desc, CUDNN_MH_ATTN_K_WEIGHTS,
                              k_num_weights * sizeof(float), size_weights, desc,
                              dev_w, k_w_data);
    init_weights<BIAS_RANK>(handle, attn_desc, CUDNN_MH_ATTN_K_BIASES,
                            k_bias_len * sizeof(float), size_weights, desc,
                            dev_w, k_b_data);

    float *v_w_data = new float[v_num_weights];
    float *v_b_data = new float[v_bias_len];
    for (size_t i = 0; i < v_num_weights; i++) {
        v_w_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < v_bias_len; i++) {
        v_b_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    init_weights<WEIGHT_RANK>(handle, attn_desc, CUDNN_MH_ATTN_V_WEIGHTS,
                              v_num_weights * sizeof(float), size_weights, desc,
                              dev_w, v_w_data);
    init_weights<BIAS_RANK>(handle, attn_desc, CUDNN_MH_ATTN_V_BIASES,
                            v_bias_len * sizeof(float), size_weights, desc,
                            dev_w, v_b_data);

    float *o_w_data = new float[o_num_weights];
    float *o_b_data = new float[o_bias_len];
    for (size_t i = 0; i < o_num_weights; i++) {
        o_w_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < o_bias_len; i++) {
        o_b_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    init_weights<WEIGHT_RANK>(handle, attn_desc, CUDNN_MH_ATTN_O_WEIGHTS,
                              o_num_weights * sizeof(float), size_weights, desc,
                              dev_w, o_w_data);
    init_weights<BIAS_RANK>(handle, attn_desc, CUDNN_MH_ATTN_O_BIASES,
                            o_bias_len * sizeof(float), size_weights, desc,
                            dev_w, o_b_data);

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

    int dimA[CUDNN_SEQDATA_DIM_COUNT];
    cudnnSeqDataAxis_t dataAxes[CUDNN_SEQDATA_DIM_COUNT];
    dataAxes[0] = CUDNN_SEQDATA_BEAM_DIM;
    dataAxes[1] = CUDNN_SEQDATA_BATCH_DIM;
    dataAxes[2] = CUDNN_SEQDATA_TIME_DIM;
    dataAxes[3] = CUDNN_SEQDATA_VECT_DIM;
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
    dimA[CUDNN_SEQDATA_TIME_DIM] = seq_len_q;
    dimA[CUDNN_SEQDATA_VECT_DIM] = embed_q;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        q_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, batch_size,
        q_seq_array, NULL));

    dimA[CUDNN_SEQDATA_VECT_DIM] = embed_o;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        o_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, batch_size,
        q_seq_array, NULL));

    dimA[CUDNN_SEQDATA_TIME_DIM] = seq_len_k;
    dimA[CUDNN_SEQDATA_VECT_DIM] = embed_k;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        k_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, batch_size,
        k_seq_array, NULL));

    dimA[CUDNN_SEQDATA_TIME_DIM] = seq_len_k;
    dimA[CUDNN_SEQDATA_VECT_DIM] = embed_v;
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        v_desc, data_type, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, batch_size,
        k_seq_array, NULL));

    float *q_data = new float[q_num_elem];
    float *k_data = new float[k_num_elem];
    float *v_data = new float[v_num_elem];
    for (size_t i = 0; i < q_num_elem; ++i) {
        q_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < k_num_elem; ++i) {
        k_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < v_num_elem; ++i) {
        v_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(dev_q, q_data, sizeof(float) * q_num_elem,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_k, k_data, sizeof(float) * k_num_elem,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_v, v_data, sizeof(float) * v_num_elem,
                          cudaMemcpyHostToDevice));

    CUDNN_CHECK(cudnnMultiHeadAttnForward(
        handle, attn_desc, -1, lo_win_idx, hi_win_idx, dev_q_seq_array,
        dev_k_seq_array, q_desc, dev_q, nullptr, k_desc, dev_k, v_desc, dev_v,
        o_desc, dev_o, size_weights, size_weights > 0 ? dev_w : nullptr,
        size_wkspace, dev_wkspace, 0, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "finished" << std::endl;

    return 0;
}
