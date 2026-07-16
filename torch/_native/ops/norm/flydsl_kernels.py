# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vendored FlyDSL plain RMSNorm forward kernel and PyTorch wrapper.

The device code is derived from ROCm/FlyDSL kernels/norm/rmsnorm_kernel.py at
commit a85595136c647b2ac4532be43ad6e37beaedc085. Only the plain RMSNorm
forward path needed by ATen is included; quantized and fused-add variants
remain out of scope.
"""

# mypy: allow-untyped-defs

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.vector import ReductionOp, full
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

from torch._native.flydsl_cache import jit_cache


_SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
_COMPILE_BACKEND_NAME = flyc.compile_backend_name()
_ROCM_ARCH_BY_DEVICE: dict[int, str] = {}
EPS = 1e-5
BLOCK_THREADS = 256
VEC_WIDTH = 8


def get_warp_size(arch=None) -> int:
    """Return wave64 for CDNA GPUs and wave32 for RDNA GPUs."""
    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64


WARP_SIZE = get_warp_size()


def _make_single_reduction_storage(red_slots: int):
    """Shared storage for one block-reduction accumulator."""

    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def dtype_to_elem_type(dtype_str: str):
    """Map the three supported PyTorch dtype strings to FlyDSL types."""
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(
        f"unsupported dtype: {dtype_str!r} "
        "(expected 'f32', 'f16', or 'bf16')"
    )


def _load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    view = fx.slice(divided_tensor, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, r)
    return fx.memref_load_vec(r)[0]


def _store_scalar(copy_atom, elem_dtype, store_dtype, divided_tensor, index, val):
    r = fx.make_rmem_tensor(1, elem_dtype)
    ts = full(1, store_dtype(val), store_dtype)
    fx.memref_store_vec(ts, r)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, r, view)


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def _to_elem_scalar(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, y):
    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return y.to(elem_dtype)
        u = y.bitcast(fx.Uint32)
        upper = u >> 16
        lsb = upper & 1
        bias = lsb + 0x7FFF
        u_round = y.bitcast(fx.Uint32) + bias
        bf16_bits = u_round >> 16
        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
        odd_sh = odd << 16
        packed = even | odd_sh
        return packed.bitcast(elem_dtype)
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _dtype_str(dtype: torch.dtype) -> str:
    try:
        return _SUPPORTED_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported RMSNorm dtype for FlyDSL: {dtype}") from exc


def _normalized_shape_1d(normalized_shape) -> int | None:
    if isinstance(normalized_shape, int):
        return normalized_shape
    if isinstance(normalized_shape, (tuple, list, torch.Size)):
        if len(normalized_shape) != 1:
            return None
        return int(normalized_shape[0])
    return int(normalized_shape)


def _compile_key_arch(device_index: int) -> str:
    arch = _ROCM_ARCH_BY_DEVICE.get(device_index)
    if arch is None:
        arch = str(get_rocm_arch())
        _ROCM_ARCH_BY_DEVICE[device_index] = arch
    return arch


def _forward_block_threads(n: int) -> int:
    if n >= 24576:
        return 1024
    if n >= 12288:
        return 512
    return BLOCK_THREADS


def build_rmsnorm_module(
    N: int,
    dtype_str: str,
    eps: float = EPS,
):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    block_threads = _forward_block_threads(N)
    elem_bits = 32 if dtype_str == "f32" else 16
    vec_width = 4 if dtype_str == "f32" else VEC_WIDTH
    tile_cols = block_threads * vec_width
    RED_SLOTS = max(1, (block_threads + WARP_SIZE - 1) // WARP_SIZE)

    SharedStorage = _make_single_reduction_storage(RED_SLOTS)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w = wave_reduce_add(val)

            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, 0.0)
                ww = wave_reduce_add(ww)

                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0)

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0):
            num_tiles = N // tile_cols
            # Layout API: buffer-backed tensors with tiled access.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(vec_width, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(vec_width, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * block_threads
                vec = _load_vec(copy_atom, vec_width, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * block_threads

                g = _load_vec(
                    copy_atom, vec_width, elem_dtype, gamma_div, idx
                ).to(fx.Float32)
                x = in_local[tile_i].to(fx.Float32)

                y = (x * rrms) * g
                out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)

                out_idx = tid + tile_i * block_threads
                _store_vec(copy_atom, vec_width, elem_dtype, out_e, out_div, out_idx)

        else:
            # ==============================================================
            # Generic path: 128-bit vector body plus scalar tail.
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            generic_vec_width = 4 if dtype_str == "f32" else VEC_WIDTH
            full_vecs = N // generic_vec_width
            vec_steps = (full_vecs + block_threads - 1) // block_threads
            scalar_tail_start = full_vecs * generic_vec_width
            scalar_tail_elems = N - scalar_tail_start

            copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b()
                if elem_bits <= 16
                else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            in_div = fx.logical_divide(row_in, fx.make_layout(generic_vec_width, 1))
            out_vec_div = fx.logical_divide(row_out, fx.make_layout(generic_vec_width, 1))
            gamma_vec_div = fx.logical_divide(Gamma_buf, fx.make_layout(generic_vec_width, 1))
            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            in_local = []
            tail_x = c_zero_f

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                is_valid = vec_idx < full_vecs
                vec_idx_safe = is_valid.select(vec_idx, 0)
                vec = _load_vec(copy_atom_v, generic_vec_width, elem_dtype, in_div, vec_idx_safe)
                in_local.append(vec)
                x = vec.to(fx.Float32)
                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2_safe = is_valid.select(red2, c_zero_f)
                thread_sumsq = thread_sumsq + red2_safe

            if const_expr(scalar_tail_elems > 0):
                tail_valid = tid < scalar_tail_elems
                tail_idx = scalar_tail_start + tid
                tail_idx_safe = tail_valid.select(tail_idx, 0)
                tail_x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, tail_idx_safe)
                tail_x = tail_x_e if dtype_str == "f32" else tail_x_e.to(fx.Float32)
                tail_x2 = tail_x * tail_x
                thread_sumsq = thread_sumsq + tail_valid.select(tail_x2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                if vec_idx < full_vecs:
                    g = _load_vec(
                        copy_atom_v, generic_vec_width, elem_dtype, gamma_vec_div, vec_idx
                    ).to(fx.Float32)
                    x = in_local[step].to(fx.Float32)
                    y = (x * rrms) * g
                    out_e = _to_elem_vec(
                        dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y
                    )
                    _store_vec(
                        copy_atom_v, generic_vec_width, elem_dtype, out_e, out_vec_div, vec_idx
                    )

            if const_expr(scalar_tail_elems > 0):
                tail_valid = tid < scalar_tail_elems
                tail_idx = scalar_tail_start + tid
                if tail_valid:
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, tail_idx)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (tail_x * rrms) * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(
                        copy_atom_s, elem_dtype, elem_dtype, out_div, tail_idx, y_e
                    )

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_kernel(Input, Gamma, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


def _make_compile_arg(tensor: torch.Tensor):
    """Make only the row dimension dynamic so one kernel supports many M values."""

    return flyc.from_torch_tensor(tensor).mark_shape_dynamic(0)


@jit_cache
def _compile_rmsnorm_fwd(
    n: int,
    dtype: str,
    eps: float,
    arch: str,
    backend: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    # arch/backend/device_index are explicit cache keys. flyc.compile binds the
    # resulting launcher to the active device/context, so cross-device reuse is
    # unsafe even when two GPUs share the same architecture.
    del arch, backend, device_index
    input_2d, weight, output_2d, rows_m, stream = compile_args
    launch = build_rmsnorm_module(n, dtype, eps=eps)
    return flyc.compile(
        launch,
        _make_compile_arg(input_2d),
        flyc.from_torch_tensor(weight),
        _make_compile_arg(output_2d),
        rows_m,
        stream,
    )


def rmsnorm_fwd(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlyDSL forward and return the ATen output/rstd pair."""
    n = _normalized_shape_1d(normalized_shape)
    if n is None:
        raise ValueError("FlyDSL RMSNorm currently requires one normalized dimension")

    rows_m = input.numel() // n
    input_shape = input.shape

    with torch.cuda.device(input.device):
        is_2d = input.ndim == 2
        input_2d = input if is_2d else input.reshape(rows_m, n)
        output_2d = torch.empty_like(input_2d)
        rstd_flat = torch.empty(rows_m, device=input.device, dtype=torch.float32)

        stream = torch.cuda.current_stream()
        device_index = input.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        compiled = _compile_rmsnorm_fwd(
            n,
            _dtype_str(input.dtype),
            float(eps),
            _compile_key_arch(device_index),
            _COMPILE_BACKEND_NAME,
            device_index,
            compile_args=(input_2d, weight, output_2d, rows_m, stream),
        )

        compiled(input_2d, weight, output_2d, rows_m, stream)

    if is_2d:
        result = output_2d, rstd_flat.view((rows_m, 1))
    else:
        stat_shape = (*input_shape[:-1], 1)
        result = output_2d.view(input_shape), rstd_flat.view(stat_shape)
    return result


def clear_rmsnorm_caches() -> None:
    """Clear native-op-level compile caches (used by tests/benchmarks)."""

    _compile_rmsnorm_fwd.cache_clear()


def rmsnorm_cache_info() -> dict[str, object]:
    """Return forward cache statistics for diagnostics."""

    return {
        "fwd": _compile_rmsnorm_fwd.cache_info(),
    }