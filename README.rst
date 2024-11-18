.. _repo: https://github.com/KLab-ai3/ai3
.. |repo| replace:: **Source Code**
.. _script: https://github.com/KLab-ai3/ai3/tree/main/bench/gather.py
.. |script| replace:: *script*
.. _custom: https://github.com/KLab-ai3/ai3/tree/main/src/ai3/custom
.. |custom| replace:: *custom*
.. _custom_cmake: https://github.com/KLab-ai3/ai3/tree/main/cmake/custom.cmake
.. |custom_cmake| replace:: *custom.cmake*
.. _model_zoo: https://github.com/KLab-ai3/ai3/tree/main/model_zoo/models.py
.. |model_zoo| replace:: *model_zoo*
.. _doc: https://klab-ai3.github.io/ai3
.. |doc| replace:: **Documentation**
.. |name| replace:: *ai3*
.. |pkg_name| replace:: *aithree*

.. _cuDNN: https://developer.nvidia.com/cudnn
.. |cuDNN| replace:: *cuDNN*
.. _SYCL: https://www.khronos.org/sycl
.. |SYCL| replace:: *SYCL*

|name|
======

The |name| (Algorithmic Innovations for Accelerated Implementations of
Artificial Intelligence) framework provides easy-to-use fine-grain algorithmic
control over an existing *DNN*. |name| contains built-in
high performance implementations of common deep learning
operations and methods by which users can implement their own algorithms in
*C++*. |name| incurs no additional performance overhead, meaning that
performance depends solely on the algorithms chosen by the user.

|doc|_ |repo|_


Framework Overview [#f1]_
"""""""""""""""""""""""""

.. figure:: https://raw.githubusercontent.com/KLab-AI3/ai3/main/docs/_static/framework_overview.png
   :align: center
   :width: 70%


Installation
""""""""""""
**From Distribution**
  1. Wheel: *pip install* |pkg_name|
  2. Source Distribution (improves library detection): *pip install* |pkg_name| *--no-binary :all:*

**From Source**
  1. Download the source code
  2. ``pip install <path to source code>``

**With Custom Implementations**
  1. Download the source code
  2. Create an implementation with the operations defined in |custom|_
  3. If needed, configure the build process with |custom_cmake|_
  4. ``pip install <path to source code>``
|name| currently features two methods for algorithmic swapping.
*convert* which converts the entire *DNN* and *swap_operation*
which swaps specific operations out of the existing *DNN*.

*swap_operation*
~~~~~~~~~~~~~
Swaps operations in-place out of the existing *DNN* for an implementation of
the user specified algorithm. After swapping, the same *DNN* can still be trained
and compiled. If no *AlgorithmicSelector* is given then the default
algorithm decided by the framework are used.

Example:
    Swaps the first *conv2d* operation for an implementation of direct convolution
    and the second *conv2d* operation for an implementation of *SMM* convolution

    >>> input_data = torch.randn(10, 3, 224, 224)
    >>> orig = ConvNet()
    >>> orig_out = orig(input_data)
    >>> ai3.swap_operation(nn.Conv2d, orig, ['direct', 'smm'])
    >>> so_out = orig(input_data)
    >>> torch.allclose(orig_out, so_out, atol=1e-6)
    True

*convert*
~~~~~~~~~~~~~~
Converts every operation in a *DNN* to an implementation of the user
specified algorithm returning a *Model* completly managed by |name|.

Algorithmic selection is performed by passing a mapping from strings
containing names of the operations to swap to a *AlgorithmicSelector*.
If no *AlgorithmicSelector* is passed for a given operation then the default
algorithm decided by the framework are used.

Example:
    Swaps the first *conv2d* operation for an implementation of direct convolution
    and the second *conv2d* operation for an implementation of *SMM* convolution

    >>> def auto_selector(orig: torch.nn.Conv2d, input_shape) -> str:
    ...     out_channels = orig.weight.shape[0]
    ...     if (out_channels < 50 and
    ...         input_shape[1] < 50 and
    ...         input_shape[2] > 150 and
    ...         input_shape[3] > 150):
    ...         return 'direct'
    ...     return 'smm'
    ...
    >>> input_data = torch.randn(1, 3, 224, 224)
    >>> vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    >>> vgg16 = vgg16.eval()
    >>> with torch.inference_mode():
    ...     torch_out = vgg16(input_data)
    ...     model: ai3.Model = ai3.convert(vgg16, {'conv2d': auto_selector,
    ...                                                 'maxpool2d': 'default'},
    ...                                         sample_input_shape=(1, 3, 224, 224))
    ...     sb_out = model(input_data)
    ...     torch.allclose(torch_out, sb_out, atol=1e-4)
    True

.. _performance:

Performance
"""""""""""
.. figure:: https://raw.githubusercontent.com/KLab-AI3/ai3/main/docs/_static/conv2d_times.png
   :alt: Latencies of Convolution Operation
   :align: center
   :width: 80%
   :figwidth: 80%

   Latency of Convolution (`details`_)

.. figure:: https://raw.githubusercontent.com/KLab-AI3/ai3/main/docs/_static/model_times.png
   :alt: Latencies of Models Relative to *PyTorch*
   :align: center
   :width: 80%
   :figwidth: 80%

   Latency of Model When Using |name| Relative to *PyTorch* (`details`_)

.. _details:

The |cudnn|_ and |sycl|_ benchmarks for both *ai3* and *PyTorch* were
gathered using an *NVIDIA GeForce L40S GPU* with *16* gigabytes of memory. The
final latencies used are the average over *10* runs after *10* warm up runs.
The implementations for the algorithms include select ones provided by *cuDNN*
and implementations from *ai3* which leverage *SYCL*. Benchmarks are
gathered using this |script|_.


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

.. [#f1] created with `draw.io <https://draw.io>`_
