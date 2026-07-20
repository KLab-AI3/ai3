*Python API*
============

.. automodule:: ai3
   :exclude-members:

Functions Providing Algorithmic Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ai3.swap_operation
.. autofunction:: ai3.convert
.. autofunction:: ai3.swap_conv2d
.. autofunction:: ai3.swap_mha

Types
~~~~~
.. autodata:: ai3.AlgorithmicSelector

Classes
~~~~~~~
.. autoclass:: ai3.Model
   :members:

.. autoclass:: ai3.Tensor
   :members:

Global Variables
~~~~~~~~~~~~~~~~
.. autodata:: ai3.SUPPORTED_ALGORITHMS
    :annotation:
.. pprint:: ai3.SUPPORTED_ALGORITHMS
.. autodata:: ai3.FROM_BACKEND
.. autodata:: ai3.SUPPORTED_FROM_BACKENDS

Other Functions
~~~~~~~~~~~~~~~
.. autofunction:: ai3.using_sycl
.. autofunction:: ai3.using_cudnn
.. autofunction:: ai3.using_cublas
.. autofunction:: ai3.using_mps_and_metal
