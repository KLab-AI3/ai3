# SPDX-License-Identifier: Apache-2.0

from collections.abc import Buffer
from typing import Sequence, Optional
from enum import Enum


def using_mps_and_metal() -> bool:
    ...


def using_cublas() -> bool:
    ...


def using_cudnn() -> bool:
    ...


def using_sycl() -> bool:
    ...


def debug_build() -> bool:
    ...


def default_opt_str() -> str:
    ...


def custom_opt_str() -> str:
    ...


def custom_mha_handles_inputs() -> bool:
    ...


def custom_mha_projects_output() -> bool:
    ...


class PaddingMode(Enum):
    zeros: int
    reflect: int
    replicate: int
    circular: int


class MHAMemFormat(Enum):
    NSE: int
    SNE: int
    NSHD: int
    SNHD: int


class ScalarType(Enum):
    Float32: int
    Float64: int


class Tensor(Buffer):
    shape: Sequence[int]
    scalar_type: ScalarType


class Model():
    def __init__(self, layers: Sequence):
        ...

    def predict(
            self, input_address: int, input_shape: Sequence[int],
            input_type: ScalarType):
        ...


def output_hw_for_2d(input: int, kernel: int,
                     padding: int,
                     dilation: Optional[int], stride):
    ...


def conv2d(input_address: int, input_shape: Sequence[int],
           input_type: ScalarType, weight_address: int,
           weight_shape: Sequence[int],
           bias_addr: Optional[int],
           padding_h: int, padding_w: int, stride_h: int, stride_w: int,
           dilation_h: int, dilation_w: int, padding_mode: PaddingMode,
           groups: int, algorithm: str) -> Tensor:
    ...


def mha(q_address: int, k_address: int, v_address: int, input_type: ScalarType,
        mem_fmt: MHAMemFormat, q_shape: Sequence[int], k_shape: Sequence[int],
        v_shape: Sequence[int], q_proj_address: int, k_proj_address: int,
        v_proj_address: int, q_bias_in_address: Optional[int],
        k_bias_in_address: Optional[int], v_bias_in_address: Optional[int],
        k_bias_address: Optional[int], v_bias_address: Optional[int],
        out_proj_address: int, out_proj_bias_address: Optional[int],
        add_zero_attn: bool, num_heads: int, k_dim: int, v_dim: int, embed_dim:
        int, dropout: float, attn_mask_address: Optional[int],
        key_padding_mask_address: Optional[int], need_weights: bool,
        average_attn_weights: bool, is_causal: bool, need_to_project: bool,
        algorithm: str) -> Tensor:
    ...


def mha_backward(dout_address: int, q_address: int, k_address: int, v_address:
                 int, input_type: ScalarType, mem_fmt: MHAMemFormat, q_shape:
                 Sequence[int], k_shape: Sequence[int], v_shape: Sequence[int],
                 q_proj_address: int, k_proj_address: int, v_proj_address: int,
                 q_bias_in_address: Optional[int], k_bias_in_address:
                 Optional[int], v_bias_in_address: Optional[int],
                 k_bias_address: Optional[int], v_bias_address: Optional[int],
                 out_proj_address: int, out_proj_bias_address: Optional[int],
                 add_zero_attn: bool, num_heads: int, k_dim: int, v_dim: int,
                 embed_dim: int, dropout: float, attn_mask_address:
                 Optional[int], key_padding_mask_address: Optional[int],
                 need_weights: bool, average_attn_weights: bool, is_causal:
                 bool, need_to_project: bool, algorithm: str) -> Tensor:
    ...


class Conv2D():
    def __init__(self,
                 weight_address: int,
                 weight_shape: Sequence[int],
                 bias_addr: Optional[int],
                 padding_h: int,
                 padding_w: int,
                 stride_h: int,
                 stride_w: int,
                 dilation_h: int,
                 dilation_w: int,
                 padding_mode: PaddingMode,
                 groups: int,
                 algorithm: str,
                 scalar_type: ScalarType):
        ...


class MaxPool2D():
    def __init__(self,
                 kernel_h: int,
                 kernel_w: int,
                 padding_h: int,
                 padding_w: int,
                 stride_h: int,
                 stride_w: int,
                 dilation_h: int,
                 dilation_w: int,
                 ceil_mode: bool,
                 algorithm: str):
        ...


class AvgPool2D():
    def __init__(self,
                 kernel_h: int,
                 kernel_w: int,
                 padding_h: int,
                 padding_w: int,
                 stride_h: int,
                 stride_w: int,
                 ceil_mode: bool,
                 count_include_pad: bool,
                 divisor_override: Optional[int],
                 algorithm: str):
        ...


class AdaptiveAvgPool2D():
    def __init__(
            self, output_h: Optional[int],
            output_w: Optional[int],
            algorithm: str):
        ...


class Linear:
    def __init__(self,
                 weight_address: int,
                 weight_shape: Sequence[int],
                 bias_addr: Optional[int],
                 algorithm: str,
                 scalar_type: ScalarType):
        ...


class ReLU():
    def __init__(self, algorithm: str):
        ...


class Flatten():
    def __init__(self, start_dim: int, end_dim: int, algorithm: str):
        ...
