// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utils.hpp>

#if defined USE_CUBLAS
#include <cublas_utils.hpp>
#endif

#if defined USE_CUDNN
#include <cudnn_utils.hpp>
#endif

#if defined USE_SYCL
#include <CL/sycl.hpp>
using namespace cl;
#endif

#if defined USE_MPS_METAL
void *gen_mps_graph_device(void);
void *gen_mps_graph(void);
void *gen_mtl_device(void);
void release_mtl_device(void *dev);
void release_mps_graph(void *g);
#endif

/**
 * @brief Initializes and caches data useful in acceleration platforms
 */
class Context {
  public:
    /**
     * @brief Returns the cached `MPSGraphDevice`, initializing it if
     * necessary
     */
#if defined USE_MPS_METAL
    inline static void *mps_graph_device() {
        if (mps_device_init) {
            return mps_g_device;
        }
        mps_g_device = gen_mps_graph_device();
        mps_device_init = true;
        return mps_g_device;
    }
#else
    inline static void *mps_graph_device() {
        errs::invalid_context_access("mps graph device", "mps");
    }
#endif

    /**
     * @brief Returns the cached `MPSGraph`, initializing it if
     * necessary
     */
#if USE_MPS_METAL
    inline static void *mps_graph() {
        if (mps_g_init) {
            return mps_g;
        }
        mps_g = gen_mps_graph();
        mps_g_init = true;
        return mps_g;
    }
#else
    inline static void *mps_graph() {
        errs::invalid_context_access("mps graph", "mps");
    }
#endif

    /**
     * @brief Returns the cached `MTLDevice`, initializing it if
     * necessary
     */
#if USE_MPS_METAL
    inline static void *mtl_device() {
        if (mtl_d_init) {
            return mtl_d;
        }
        mtl_d = gen_mtl_device();
        mtl_d_init = true;
        return mtl_d;
    }
#else
    inline static void *mtl_device() {
        errs::invalid_context_access("metal device", "metal");
    }
#endif

    /**
     * @brief Returns the cached `cudnnHandle_t`, initializing it if
     * necessary
     */
#if defined USE_CUDNN
    inline static void *cudnn_handle_t() {
        if (cudnn_init) {
            return cudnn_handle;
        }
        CUDNN_CHECK(cudnnCreate(&cudnn_handle));
        cudnn_init = true;
        return cudnn_handle;
    }
#else
    inline static void *cudnn_handle_t() {
        errs::invalid_context_access("cudnn handle", "cudnn");
    }
#endif

    /**
     * @brief Returns the cached `cublasHandle_t`, initializing it if
     * necessary
     */
#if defined USE_CUBLAS
    inline static void *cublas_handle_t() {
        if (cublas_init) {
            return cublas_handle;
        }
        CUBLAS_CHECK(cublasCreate(&cublas_handle))
        cublas_init = true;
        return cublas_handle;
    }
#else
    inline static void *cublas_handle_t() {
        errs::invalid_context_access("cublas handle", "cublas");
    }
#endif

    /**
     * @brief Returns the cached `sycl::queue`, initializing it if
     * necessary
     */
#if defined USE_SYCL
    inline static void *sycl_queue() {
        if (sycl_init) {
            return sycl_q;
        }
        sycl_q = new sycl::queue(sycl::default_selector_v);
        sycl_init = true;
        return sycl_q;
    }
#else
    inline static void *sycl_queue() {
        errs::invalid_context_access("sycl queue", "sycl");
    }
#endif

    ~Context() {
#if defined USE_CUDNN
        if (cudnn_init) {
            CUDNN_CHECK(cudnnDestroy(cudnn_handle));
        }
#endif

#if defined USE_CUBLAS
        if (cublas_init) {
            CUBLAS_CHECK(cublasDestroy(cublas_handle));
        }
#endif
#if defined USE_MPS_METAL
        if (mps_g_init) {
            release_mps_graph(mps_g);
        }
        if (mtl_d_init) {
            release_mtl_device(mtl_d);
        }
#endif
    }

  private:
#if defined USE_MPS_METAL
    inline static void *mps_g_device = nullptr;
    inline static bool mps_device_init = false;
    inline static void *mps_g = nullptr;
    inline static bool mps_g_init = false;
    inline static void *mtl_d = nullptr;
    inline static bool mtl_d_init = false;
#endif

#if defined USE_CUDNN
    inline static cudnnHandle_t cudnn_handle;
    inline static bool cudnn_init = false;
#endif
#if defined USE_CUBLAS
    inline static cublasHandle_t cublas_handle;
    inline static bool cublas_init = false;
#endif

#if defined USE_SYCL
    inline static sycl::queue *sycl_q = nullptr;
    inline static bool sycl_init = false;
#endif
};
