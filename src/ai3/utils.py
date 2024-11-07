# SPDX-License-Identifier: Apache-2.0

from typing import (
    Union,
    Sequence,
    Any,
    Type,
    Tuple,
    Mapping,
    Callable,
    Optional,
)
import inspect
from . import errors, _core

FLOAT32_STR = 'float32'
FLOAT64_STR = 'float64'

TORCH_TENSOR_TYPE_STR = 'torch.Tensor'
TORCH_MODULE_TYPE_STR = 'torch.nn.modules.module.Module'

AlgorithmicSelector = Union[str, Sequence[str], Callable]

SUPPORTED_ALGORITHMS = {
    'conv2d': ['direct', 'smm', 'winograd', 'gemm',
               'implicit gemm', 'implicit precomp gemm',
               'guess', 'mps', 'metal'],
    'mha': [],
    'linear': ['gemm'],
    'maxpool2d': ['direct'],
    'avgpool2d': ['direct'],
    'adaptiveavgpool2d': ['direct'],
    'linear': ['direct'],
    'relu': ['direct'],
    'flatten': ['direct'],
}
DEFAULT_OPTION = _core.default_opt_str()
for key in SUPPORTED_ALGORITHMS:
    SUPPORTED_ALGORITHMS[key].append(DEFAULT_OPTION)


def check_callable_params_with_shape(
        algos: Mapping[str, Union[str, Sequence[str],
                                  Callable]],
        sample_input_shape: Optional[Sequence[int]] = None):
    for key, value in algos.items():
        if callable(value):
            if len(inspect.signature(value).parameters) == 2:
                errors.bail_if(
                    sample_input_shape is None,
                    f'for {key} using function selector which depends on input shape, but no sample input shape provided')
            else:
                errors.bail_if(
                    sample_input_shape is not None,
                    f'provided sample input shape but function selector for {key} doesn\'t take an input shape')


def get_scalar_type(dtype) -> _core.ScalarType:
    if str(dtype) == 'torch.float32':
        return _core.ScalarType(_core.ScalarType.Float32)
    if str(dtype) == 'torch.float64':
        return _core.ScalarType(_core.ScalarType.Float64)
    errors.bail(f'using bad dtype: {str(dtype)}')


def get_swapper(
        orig, from_backend: Optional[str],
        supported_backends: Sequence[str]):
    if not from_backend:
        type_str = smart_type_str(type(orig))
        if type_str == TORCH_MODULE_TYPE_STR:
            from_backend = 'torch'
        else:
            errors.bail(
                f'algorithmic selection is not suported for type {type_str}')
    try:
        if from_backend == 'torch':
            from . import swap_torch
            return swap_torch
    except ModuleNotFoundError as e:
        errors.print_error(
            f'using backend {from_backend}, but it was not found')
        raise e
    errors.bail(
        f'unsupported backend: {from_backend}, supported backends are: {supported_backends}')


def smart_type_str(orig_type: Union[Type, str]) -> str:
    def is_subclass_of(orig_type, type_str):
        return any(
            base.__module__ + '.' + base.__name__ == type_str
            for base in orig_type.__mro__)
    if isinstance(orig_type, str):
        return orig_type
    if is_subclass_of(orig_type, TORCH_TENSOR_TYPE_STR):
        return TORCH_TENSOR_TYPE_STR
    if is_subclass_of(orig_type, TORCH_MODULE_TYPE_STR):
        return TORCH_MODULE_TYPE_STR
    return f'{orig_type.__module__}.{orig_type.__name__}'


def get_address(frontend_data) -> int:
    if smart_type_str(type(frontend_data)) == TORCH_TENSOR_TYPE_STR:
        return frontend_data.data_ptr()
    errors.bail(
        f'bad input type {type(frontend_data)} when getting data address')


def get_shape(frontend_data) -> tuple:
    if smart_type_str(type(frontend_data)) == TORCH_TENSOR_TYPE_STR:
        return frontend_data.size()
    assert False and 'bad input type when getting shape'


def make_padding_2d(padding: Union[str, Union[int, Tuple[int, ...]]],
                    stride: Tuple[int, int], dilation: Tuple[int, int],
                    kernel_shape: Sequence[int], dtype=int) -> tuple[int, int]:
    if isinstance(padding, str):
        if padding == 'valid':
            return (0, 0)
        if padding == 'same':
            assert all(x == 1 for x in stride)
            k_height = kernel_shape[len(
                kernel_shape) - 2]
            k_width = kernel_shape[len(
                kernel_shape) - 1]
            return (dilation[0] * (k_height - 1) // 2, dilation[1] * (k_width - 1) // 2)
        assert False and f'invalid padding string: {padding}'
    else:
        return make_2d(padding, dtype=dtype)


def make_2d(a: Union[Any, Tuple[Any, Any]],
            dtype: type = int) -> Tuple[Any, Any]:
    if isinstance(a, dtype):
        return (a, a)
    assert len(a) == 2 and all(
        isinstance(val, dtype) for val in a)
    return a
