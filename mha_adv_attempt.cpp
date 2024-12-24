#include <cudnn.h>
#include <iostream>

#define CUDA_CHECK(status)                                                     \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(status)              \
                  << std::endl;                                                \
        exit(1);                                                               \
    }

#define CUDNN_CHECK(status)                                                    \
    {                                                                          \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudnnGetErrorString(status) << std::endl;     \
            std::cerr << std::endl;                                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    }

int main() {
    int batch_size = 2;
    int seq_len = 10;
    int embed_dim = 64;
    int num_heads = 4;
    int proj_dim = embed_dim / num_heads;

    int q_shape[3] = {batch_size, seq_len, embed_dim};
    int k_shape[3] = {batch_size, seq_len, embed_dim};
    int v_shape[3] = {batch_size, seq_len, embed_dim};

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnDropoutDescriptor_t attn_dropout_desc;
    cudnnDropoutDescriptor_t post_dropout_desc;
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&attn_dropout_desc));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&post_dropout_desc));

    size_t states_size;
    void *states;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &states_size));
    cudaMalloc(&states, states_size);

    CUDNN_CHECK(cudnnSetDropoutDescriptor(attn_dropout_desc, handle, 0.0,
                                          states, states_size, 0));
    CUDNN_CHECK(cudnnSetDropoutDescriptor(post_dropout_desc, handle, 0.0,
                                          states, states_size, 0));

    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnAttnDescriptor_t attn_desc;
    CUDNN_CHECK(cudnnCreateAttnDescriptor(&attn_desc));
    CUDNN_CHECK(cudnnSetAttnDescriptor(
        attn_desc, CUDNN_ATTN_QUERYMAP_ONE_TO_ONE, num_heads, 1.0, data_type,
        CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, attn_dropout_desc,
        post_dropout_desc, embed_dim, embed_dim, embed_dim, proj_dim, proj_dim,
        proj_dim, 0, seq_len, seq_len, batch_size, 1));

    cudnnSeqDataDescriptor_t q_desc;
    cudnnSeqDataDescriptor_t k_desc;
    cudnnSeqDataDescriptor_t v_desc;
    cudnnSeqDataDescriptor_t o_desc;
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&q_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&k_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&v_desc));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&o_desc));
    cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
    // TODO do we need NHWC or can stay with nchw
    axes[0] = CUDNN_SEQDATA_BATCH_DIM;
    axes[1] = CUDNN_SEQDATA_TIME_DIM;
    axes[2] = CUDNN_SEQDATA_BEAM_DIM;
    axes[3] = CUDNN_SEQDATA_VECT_DIM;

    int dimA[CUDNN_SEQDATA_DIM_COUNT];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
    dimA[CUDNN_SEQDATA_TIME_DIM] = num_heads;
    dimA[CUDNN_SEQDATA_BEAM_DIM] = seq_len;
    dimA[CUDNN_SEQDATA_VECT_DIM] = proj_dim;

    int length_array_size = proj_dim * seq_len;
    int q_array_size = testCfg.beamSize * testCfg.batchSize;
    int k_array_size = testCfg.batchSize;

    // TODO account for not self-attention
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(q_desc, data_type,
                                          CUDNN_SEQDATA_DIM_COUNT, dimA, axes,
                                          qSeqArraySize, qSeqArray, NULL));

    size_t maxWeights = 0;
    size_t maxWkspace = 0;
    CUDNN_CHECK(cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &maxWeights,
                                             &maxWkspace, nullptr));

    std::cout << "done" << std::endl;

    return 0;
}
