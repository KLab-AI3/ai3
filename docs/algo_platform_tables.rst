Supported Operations, their Algorithms, and Acceleration Platform Compatibility
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. |y| unicode:: U+2713
.. |n| unicode:: U+2717

*2D* Convolution
~~~~~~~~~~~~~~~~

The *guess* algorithm uses the algorithm returned by `cudnnGetConvolutionForwardAlgorithm_v7`.

.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
     - *smm*
     - *gemm*
     - *implicit precomp gemm*
     - *implicit gemm*
     - *winograd*
     - *guess*
     - some
   * - *none*
     - |y|
     - |y|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |y|
   * - *sycl*
     - |y|
     - |y|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |y|
   * - *cudnn*
     - |n|
     - |n|
     - |y|
     - |y|
     - |y|
     - |y|
     - |y|
     - |y|
   * - *cublas*
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
   * - *mps*
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |y|
   * - *metal*
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |y|

Linear
~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - *gemm*
   * - *none*
     - |y|
   * - *sycl*
     - |y|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |y|
   * - *mps*
     - |n|
   * - *metal*
     - |n|


Multihead Attention
~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - *standard*
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |y|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|


*2D* MaxPool
~~~~~~~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|

*2D* AvgPool
~~~~~~~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|

*2D* AdaptiveAvgPool
~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|

*ReLU*
~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|


Flatten
~~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|
