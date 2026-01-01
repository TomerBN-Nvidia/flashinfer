#!/usr/bin/env python3
"""
================================================================================
FlashInfer Mixed Precision 4-bit × 8-bit MoE Error-Surfacing Script
(DeepSeek Block-Scale FP8 Quantization Version)
================================================================================

PURPOSE:
    This script explicitly attempts to run 4-bit weight × 8-bit activation
    mixed-precision MoE configurations using FlashInfer's cutlass_fused_moe().
    
    The goal is to SURFACE ERRORS and UNDERSTAND FAILURE MODES, not to make
    operations succeed. This helps validate kernel support assumptions and
    understand how FlashInfer/CUTLASS rejects unsupported configurations.

DIFFERENCE FROM test_mixed_precision_4bit_8bit_moe.py:
    This version uses DeepSeek-style BLOCK-SCALE FP8 quantization for activations
    instead of per-tensor quantization. Block-scale quantization uses 128-element
    blocks where each block has its own scale factor.

TARGET HARDWARE:
    - Hopper GPU (SM >= 90)
    
CONFIGURATIONS TESTED:
    1. NVFP4 weights × FP8 activations (block-scale)
       - Expected: FAIL on SM90 (NVFP4 requires SM100+)
       
    2. MXFP4 weights × FP8 activations (block-scale)
       - Expected: FAIL on SM90 (MXFP4×FP8 requires SM100+)
       
    3. Int4 weights × FP8 activations (W4A8 with block-scale)
       - Expected: May work on SM90 via W4A8 path with use_w4_group_scaling=True

MoE STRUCTURE:
    - Non-gated MoE (ActivationType.Relu2)
    - Routing is precomputed (no gating logic in kernel)
    - Output dtype: BF16

BLOCK-SCALE QUANTIZATION:
    - Block size: 128 elements per block
    - Scale shape: [num_tokens, hidden_size // 128] for activations
    - Transpose layout: [hidden_size // 128, num_tokens] for some APIs

⚠️ IMPORTANT:
    This script intentionally does NOT:
    - Fall back to supported dtypes
    - Skip unsupported configurations
    - Convert weights/activations to higher precision automatically
    
Reference:
    - Quantization POR validation
    - Scope NVFP4 / MXFP4 / Int4 × FP8 GEMM for SM >= 90
    - DeepSeek block-scale FP8 quantization pattern
    
================================================================================
"""

import traceback
import sys
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


# ==============================================================================
# Configuration Classes
# ==============================================================================

class WeightFormat(Enum):
    """4-bit weight format types."""
    NVFP4 = "nvfp4"      # NVIDIA FP4 (E2M1 format)
    MXFP4 = "mxfp4"      # Microscaling FP4
    INT4 = "int4"         # Integer 4-bit


@dataclass
class MoEConfig:
    """MoE test configuration."""
    num_tokens: int = 4
    hidden_size: int = 128
    intermediate_size: int = 128
    num_experts: int = 4
    top_k: int = 2
    output_dtype: torch.dtype = torch.bfloat16
    weight_format: WeightFormat = WeightFormat.NVFP4


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_device_info() -> Dict[str, Any]:
    """Get GPU device information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    
    return {
        "device_name": props.name,
        "compute_capability": f"{major}.{minor}",
        "sm_version": major * 10 + minor,
        "total_memory_gb": props.total_memory / (1024**3),
        "is_hopper": major == 9,
        "is_blackwell": major >= 10,
    }


def compute_routing(
    router_logits: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.
    
    Args:
        router_logits: Router logits of shape [batch_size, num_experts]
        top_k: Number of experts to route to per token

    Returns:
        routing_weights: Expert weights of shape [batch_size, top_k]
        selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def fp8_block_quant_1d(
    x_bf16: torch.Tensor, 
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek-style block-scale FP8 quantization for activations.
    
    Quantize [T, H] activations into FP8 with per-(token, 128-col) block scales.
    
    Args:
        x_bf16: Input tensor of shape [T, H] in BF16
        block_size: Block size for quantization (default: 128)
    
    Returns:
        x_fp8: Quantized tensor [T, H] in float8_e4m3fn
        scales: Dequant scales [T, H/block_size] in float32 (float ≈ fp8 * scale)
    """
    assert x_bf16.dim() == 2
    T, H = x_bf16.shape
    assert H % block_size == 0, f"H={H} must be divisible by block_size={block_size}"
    nb = H // block_size

    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max

    x_f32 = x_bf16.to(torch.float32)
    x_fp8 = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=x_bf16.device)
    scales = torch.empty((T, nb), dtype=torch.float32, device=x_bf16.device)

    for j in range(nb):
        sl = slice(j * block_size, (j + 1) * block_size)
        blk = x_f32[:, sl]  # [T, block_size]
        amax = torch.amax(torch.abs(blk), dim=1)  # [T]
        # dequant scale s = amax / max_fp8  (float ≈ fp8 * s)
        s = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
        q = (blk / s.unsqueeze(1)).to(torch.float8_e4m3fn)  # quantization
        x_fp8[:, sl] = q
        scales[:, j] = s
    
    return x_fp8, scales  # scales in [T, H/block_size]


def fp8_block_quant_2d(
    w_bf16: torch.Tensor, 
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-scale FP8 quantization for weights with 2D blocking.
    
    Quantize weights with 2D block scales over the last two dims.
    
    Args:
        w_bf16: Weight tensor [*, R, C] where R and C are multiples of block_size
        block_size: Block size for quantization (default: 128)
    
    Returns:
        w_fp8: Quantized tensor [*, R, C] in float8_e4m3fn
        scales: Dequant scales [*, R/block_size, C/block_size] in float32
    """
    assert w_bf16.dim() >= 2
    *prefix, R, C = w_bf16.shape
    assert R % block_size == 0 and C % block_size == 0
    nb_r = R // block_size
    nb_c = C // block_size

    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max

    w_f32 = w_bf16.to(torch.float32).contiguous()
    prefix_ndim = len(prefix)

    # Reshape weights into block_size x block_size blocks and move block dims to the tail:
    # [..., nb_r, block_size, nb_c, block_size] -> [..., nb_r, nb_c, block_size, block_size]
    reshaped = w_f32.reshape(*prefix, nb_r, block_size, nb_c, block_size)
    permute_dims = tuple(range(prefix_ndim)) + (
        prefix_ndim,
        prefix_ndim + 2,
        prefix_ndim + 1,
        prefix_ndim + 3,
    )
    blocks = reshaped.permute(permute_dims).contiguous()

    # Compute per-block scales
    amax = torch.amax(torch.abs(blocks), dim=(-1, -2))
    scales = torch.where(
        amax > 0,
        amax / max_fp8,
        torch.ones_like(amax, dtype=torch.float32),
    )

    # Quantize blocks in parallel
    q_blocks = (blocks / scales.unsqueeze(-1).unsqueeze(-1)).to(torch.float8_e4m3fn)

    # Restore original layout
    inv_permute = [0] * (prefix_ndim + 4)
    for i, d in enumerate(permute_dims):
        inv_permute[d] = i
    w_fp8 = q_blocks.permute(*inv_permute).reshape(*prefix, R, C)

    return w_fp8, scales


def print_separator(title: str = None):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'-'*80}\n")


def print_config(config: MoEConfig):
    """Print MoE configuration."""
    print(f"  Weight Format:      {config.weight_format.value}")
    print(f"  Activation Format:  FP8 (E4M3) with DeepSeek Block-Scale (128)")
    print(f"  Output dtype:       {config.output_dtype}")
    print(f"  Num Tokens:         {config.num_tokens}")
    print(f"  Hidden Size:        {config.hidden_size}")
    print(f"  Intermediate Size:  {config.intermediate_size}")
    print(f"  Num Experts:        {config.num_experts}")
    print(f"  Top-K:              {config.top_k}")
    print(f"  Activation Type:    Relu2 (non-gated)")


# ==============================================================================
# Test Functions for Each Weight Format with Block-Scale FP8
# ==============================================================================

def test_nvfp4_x_fp8_blockscale(config: MoEConfig) -> Dict[str, Any]:
    """
    Test NVFP4 (4-bit) weights × FP8 (8-bit) activations with block-scale quantization.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration should FAIL because NVFP4 is only supported on 
        Blackwell architecture (SM100+). The error should occur at:
        - Module initialization (JIT compilation)
        - Or kernel selection phase
        
    The NVFP4 format uses E2M1 (2-bit exponent, 1-bit mantissa) which is 
    a native format on Blackwell but not Hopper.
    
    Block-scale FP8 quantization:
        - Activations are quantized with 128-element blocks
        - Each block has its own scale factor
        - Scale shape: [T, H/128] or transposed [H/128, T]
    """
    print_separator("TEST: NVFP4 × FP8 MoE (Block-Scale)")
    print_config(config)
    print()
    
    result = {
        "config": "NVFP4 × FP8 (Block-Scale)",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        # Import FlashInfer modules
        # This may fail if NVFP4 kernels are not available for SM90
        import flashinfer
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        from flashinfer import fp4_quantize
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        # Create test tensors
        m = config.num_tokens
        k = config.hidden_size
        n = config.intermediate_size
        e = config.num_experts
        top_k = config.top_k
        quant_blocksize = 16
        
        # Round up helper
        round_up = lambda x, y: (x + y - 1) // y * y
        
        # Create high-precision weights first
        # For Relu2 (non-gated), w1 has shape [e, n, k] (not 2*n)
        w1 = torch.randn((e, n, k), device=device, dtype=torch.bfloat16) / 10
        w2 = torch.randn((e, k, n), device=device, dtype=torch.bfloat16) / 10
        
        # Create FP8 activations with BLOCK-SCALE quantization
        x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        x_fp8, x_scales = fp8_block_quant_1d(x_bf16, block_size=128)
        # Transpose scales for API compatibility: [T, H/128] -> [H/128, T]
        x_scales_transposed = x_scales.t().contiguous()
        
        print(f"\n  Created FP8 activations with block-scale quantization:")
        print(f"    x_fp8 shape: {x_fp8.shape}, dtype: {x_fp8.dtype}")
        print(f"    x_scales shape: {x_scales.shape} (original)")
        print(f"    x_scales_transposed shape: {x_scales_transposed.shape} (for API)")
        
        # Quantize weights to NVFP4
        # NVFP4 uses fp4_quantize which packs two FP4 values per uint8
        FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
        FLOAT4_E2M1_MAX = 6.0
        
        # Compute block scales for weights
        sf_w1_n = round_up(n, 128)
        sf_w1_k = round_up(k // quant_blocksize, 4)
        sf_w2_k = round_up(k, 128)
        sf_w2_n = round_up(n // quant_blocksize, 4)
        
        w1_q = torch.empty((e, n, k // 2), device=device, dtype=torch.uint8)
        w2_q = torch.empty((e, k, n // 2), device=device, dtype=torch.uint8)
        w1_blockscale = torch.empty((e, sf_w1_n, sf_w1_k), device=device, dtype=torch.float8_e4m3fn)
        w2_blockscale = torch.empty((e, sf_w2_k, sf_w2_n), device=device, dtype=torch.float8_e4m3fn)
        w1_gs = torch.empty((e,), device=device, dtype=torch.float32)
        w2_gs = torch.empty((e,), device=device, dtype=torch.float32)
        
        print(f"  Quantizing weights to NVFP4...")
        
        for expert in range(e):
            w1_amax = torch.abs(w1[expert]).max().to(torch.float32)
            w2_amax = torch.abs(w2[expert]).max().to(torch.float32)
            w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
            w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
            
            # Quantize to FP4
            # This call may fail on SM90
            w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])
            w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])
        
        print(f"  NVFP4 weights created: w1_q shape={w1_q.shape}, dtype={w1_q.dtype}")
        
        # Create routing
        router_logits = torch.randn(m, e, dtype=torch.bfloat16, device=device)
        routing_weights, selected_experts = compute_routing(router_logits, top_k)
        
        # Prepare quant_scales for NVFP4 with block-scale FP8 activations
        # Format: [fc1_act_global, fc1_weight_block, fc1_global, fc2_act_global, fc2_weight_block, fc2_global]
        a1_gs = torch.tensor(1.0, device=device, dtype=torch.float32)
        a2_gs = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        quant_scales = [
            a1_gs,                                    # fc1 activation global scale
            w1_blockscale.view(torch.int32),          # fc1 weight block scales
            1.0 / (a1_gs * w1_gs),                    # fc1 dequant scale
            a2_gs,                                    # fc2 activation global scale
            w2_blockscale.view(torch.int32),          # fc2 weight block scales
            1.0 / (a2_gs * w2_gs),                    # fc2 dequant scale
        ]
        
        # Prepare output tensor
        output = torch.zeros_like(x_bf16)
        
        print(f"\n  Calling cutlass_fused_moe with NVFP4 weights × FP8 block-scale activations...")
        print(f"  ⚠️  This is expected to FAIL on SM90 (Hopper)")
        print()
        
        # The key call - attempting NVFP4 × FP8 (block-scale) MoE
        # Note: Using use_deepseek_fp8_block_scale=True to indicate block-scale activation
        _ = cutlass_fused_moe(
            x_fp8,  # FP8 activations (8-bit) with block-scale
            selected_experts.to(torch.int),
            routing_weights,
            w1_q.contiguous().view(torch.long),  # NVFP4 weights viewed as int64
            w2_q.contiguous().view(torch.long),  # NVFP4 weights viewed as int64
            config.output_dtype,
            quant_scales=quant_scales,
            input_sf=x_scales_transposed,  # Block-scale factors for activation
            use_deepseek_fp8_block_scale=True,  # Enable block-scale mode
            output=output,
            activation_type=ActivationType.Relu2,  # Non-gated
        )
        
        # If we get here, it unexpectedly succeeded
        result["success"] = True
        print("  ✓ UNEXPECTED SUCCESS: NVFP4 × FP8 (block-scale) completed without error!")
        
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        
        # Determine where the error occurred
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_frame = tb[-1]
            result["error_location"] = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        
        print(f"  ✗ ERROR CAPTURED (as expected on SM90):")
        print(f"    Type: {result['error_type']}")
        print(f"    Message: {result['error_message']}")
        print(f"    Location: {result['error_location']}")
        print(f"\n  Full Traceback:")
        print("  " + result["traceback"].replace("\n", "\n  "))
    
    return result


def test_mxfp4_x_fp8_blockscale(config: MoEConfig) -> Dict[str, Any]:
    """
    Test MXFP4 (4-bit) weights × FP8 (8-bit) activations with block-scale quantization.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration should FAIL because MXFP4×FP8 requires Blackwell 
        architecture (SM100+). The kernel instantiation for this combination
        is gated behind ENABLE_FP4 which is only defined for SM100+.
        
    MXFP4 uses microscaling with FP8 E8M0 scale factors per 32-element block.
    
    Block-scale FP8 quantization for activations:
        - Activations are quantized with 128-element blocks
        - Each block has its own scale factor
    """
    print_separator("TEST: MXFP4 × FP8 MoE (Block-Scale)")
    print_config(config)
    print()
    
    result = {
        "config": "MXFP4 × FP8 (Block-Scale)",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        from flashinfer import mxfp4_quantize
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m = config.num_tokens
        k = config.hidden_size
        n = config.intermediate_size
        e = config.num_experts
        top_k = config.top_k
        
        # Create high-precision weights
        # For Relu2 (non-gated), use regular shape without gating dimension
        w1 = torch.randn((e, 2 * n, k), device=device, dtype=torch.bfloat16) / 10  # Keep gated shape for API compatibility
        w2 = torch.randn((e, k, n), device=device, dtype=torch.bfloat16) / 10
        
        # Create input and quantize to FP8 with BLOCK-SCALE
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        
        print(f"  Quantizing activations to FP8 with block-scale (128)...")
        x_fp8, x_scales = fp8_block_quant_1d(x, block_size=128)
        # Transpose scales for API compatibility: [T, H/128] -> [H/128, T]
        x_scales_transposed = x_scales.t().contiguous()
        
        print(f"  FP8 block-scale activations:")
        print(f"    x_fp8 shape: {x_fp8.shape}, dtype: {x_fp8.dtype}")
        print(f"    x_scales shape: {x_scales.shape} -> transposed: {x_scales_transposed.shape}")
        
        print(f"  Quantizing weights to MXFP4...")
        
        # Quantize weights to MXFP4
        def quant_mxfp4_batches(a, num_experts):
            quant_a = []
            sfs = []
            for i in range(num_experts):
                a_fp4, a_sf = mxfp4_quantize(a[i].cuda())
                quant_a.append(a_fp4)
                sfs.append(a_sf)
            return torch.stack(quant_a), torch.stack(sfs)
        
        mxfp4_w1, mxfp4_w1_scale = quant_mxfp4_batches(w1, e)
        mxfp4_w2, mxfp4_w2_scale = quant_mxfp4_batches(w2, e)
        
        print(f"  MXFP4 weights: w1 shape={mxfp4_w1.shape}, dtype={mxfp4_w1.dtype}")
        
        # Create routing
        router_logits = torch.randn(m, e, dtype=torch.bfloat16, device=device)
        routing_weights, selected_experts = compute_routing(router_logits, top_k)
        
        # Prepare quant_scales for MXFP4 × FP8 block-scale
        # Format: [w1_scale, fake_input_scale, w2_scale, fake_input_scale]
        fake_input_scale = torch.ones(e, device=device)
        
        quant_scales = [
            mxfp4_w1_scale.view(torch.int32),
            fake_input_scale,
            mxfp4_w2_scale.view(torch.int32),
            fake_input_scale,
        ]
        
        output = torch.zeros_like(x)
        
        print(f"\n  Calling cutlass_fused_moe with MXFP4 weights × FP8 block-scale activations...")
        print(f"  ⚠️  This is expected to FAIL on SM90 (Hopper)")
        print()
        
        # The key call - attempting MXFP4 × FP8 (block-scale) MoE
        _ = cutlass_fused_moe(
            x_fp8,  # FP8 activations (block-scale quantized)
            selected_experts.to(torch.int),
            routing_weights,
            mxfp4_w1.contiguous().view(torch.long),  # MXFP4 weights
            mxfp4_w2.contiguous().view(torch.long),  # MXFP4 weights
            config.output_dtype,
            quant_scales=quant_scales,
            input_sf=x_scales_transposed,  # Block-scale factors [H/128, T]
            use_deepseek_fp8_block_scale=True,  # Enable block-scale mode
            output=output,
            activation_type=ActivationType.Relu2,  # Non-gated
        )
        
        result["success"] = True
        print("  ✓ UNEXPECTED SUCCESS: MXFP4 × FP8 (block-scale) completed without error!")
        
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_frame = tb[-1]
            result["error_location"] = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        
        print(f"  ✗ ERROR CAPTURED (as expected on SM90):")
        print(f"    Type: {result['error_type']}")
        print(f"    Message: {result['error_message']}")
        print(f"    Location: {result['error_location']}")
        print(f"\n  Full Traceback:")
        print("  " + result["traceback"].replace("\n", "\n  "))
    
    return result


def test_int4_x_fp8_blockscale(config: MoEConfig) -> Dict[str, Any]:
    """
    Test Int4 (4-bit) weights × FP8 (8-bit) activations (W4A8 path) with block-scale.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration MAY SUCCEED on SM90 because the W4A8 path is 
        explicitly supported via use_w4_group_scaling=True. This uses
        FP8 E4M3 activations with INT4 packed weights.
        
    The Int4 format uses signed 4-bit integers with per-group scaling.
    
    NOTE: W4A8 uses Swiglu (gated) activation, not Relu2. The gated path
    requires fc1 to have 2x intermediate_size for the gate+up projection.
    
    Block-scale FP8 quantization:
        - Activations are quantized with 128-element blocks
        - Each block has its own scale factor
    """
    print_separator("TEST: Int4 × FP8 MoE (W4A8 + Block-Scale)")
    print_config(config)
    print()
    
    result = {
        "config": "Int4 × FP8 (W4A8 + Block-Scale)",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m = config.num_tokens
        k = config.hidden_size
        n = config.intermediate_size
        e = config.num_experts
        top_k = config.top_k
        group_size = 128
        affine_coeff = 0.005
        
        # Create input and quantize to FP8 with BLOCK-SCALE
        x = torch.randn(m, k, dtype=config.output_dtype, device=device)
        x_fp8, x_scales = fp8_block_quant_1d(x, block_size=128)
        # Transpose scales for API compatibility: [T, H/128] -> [H/128, T]
        x_scales_transposed = x_scales.t().contiguous()
        
        print(f"  Created FP8 block-scale activations:")
        print(f"    x_fp8 shape: {x_fp8.shape}, dtype: {x_fp8.dtype}")
        print(f"    x_scales shape: {x_scales.shape} -> transposed: {x_scales_transposed.shape}")
        
        # =======================================================================
        # W4A8 uses GATED activation (Swiglu), so fc1 needs 2x intermediate_size
        # Shape requirements for W4A8:
        #   - fc1: [num_experts, 2 * intermediate_size, hidden_size // 2]
        #   - fc2: [num_experts, hidden_size, intermediate_size // 2]
        # The check: fc1.size(1) == fc2.size(2) * mInnerDimMultiplier
        #   where mInnerDimMultiplier = 2 for INT4
        # =======================================================================
        
        # Create INT4 quantized weights (packed as 2 values per uint8)
        # w1 and w3 are for the gated projection, w2 is for output projection
        w1_weight = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device=device)
        w2_weight = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, device=device)
        w3_weight = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device=device)
        
        print(f"  Created INT4 weights:")
        print(f"    w1 (up proj):   shape={w1_weight.shape}, dtype={w1_weight.dtype}")
        print(f"    w3 (gate proj): shape={w3_weight.shape}, dtype={w3_weight.dtype}")
        print(f"    w2 (down proj): shape={w2_weight.shape}, dtype={w2_weight.dtype}")
        
        # Per-group weight scales
        w1_scale = torch.randn(e, n, k // group_size, dtype=config.output_dtype, device=device) * affine_coeff
        w2_scale = torch.randn(e, k, n // group_size, dtype=config.output_dtype, device=device) * affine_coeff
        w3_scale = torch.randn(e, n, k // group_size, dtype=config.output_dtype, device=device) * affine_coeff
        
        # Per-channel pre-quant scales
        w1_pre_quant_scale = torch.rand(e, k, dtype=config.output_dtype, device=device) * 0.1 + 0.95
        w2_pre_quant_scale = torch.rand(e, n, dtype=config.output_dtype, device=device) * 0.1 + 0.95
        w3_pre_quant_scale = torch.rand(e, k, dtype=config.output_dtype, device=device) * 0.1 + 0.95
        
        input_scale = torch.rand(e, 1, dtype=torch.float32, device=device) * 0.2 + 0.1
        weight_scale_2 = torch.ones(e, 1, dtype=torch.float32, device=device)
        
        # Combine w3 and w1 weights for gated projection: [e, 2*n, k//2]
        fc1_weights = torch.cat([w3_weight, w1_weight], dim=1)
        fc2_weights = w2_weight
        
        print(f"\n  Combined weights for API:")
        print(f"    fc1 (gated): shape={fc1_weights.shape} [e, 2*n, k//2]")
        print(f"    fc2:         shape={fc2_weights.shape} [e, k, n//2]")
        print(f"    Check: fc1.size(1)={fc1_weights.size(1)} == fc2.size(2)*2={fc2_weights.size(2)*2}")
        
        # Interleave weights for TRTLLM format
        def interleave_weights(w: torch.Tensor, dim: int) -> torch.Tensor:
            interleave_factor = 4 if dim % 512 == 0 else (2 if dim % 256 == 0 else 1)
            s = w.shape
            w_interleaved = (
                w.reshape(s[0], s[1], s[2] // interleave_factor, interleave_factor)
                .permute(0, 2, 1, 3)
                .reshape(s[0], s[2] // interleave_factor, s[1] * interleave_factor)
                .contiguous()
            )
            return w_interleaved
        
        w3_w1_scales = torch.cat([w3_scale, w1_scale], dim=1)
        w3_w1_scales_int = interleave_weights(w3_w1_scales, k)
        w2_scales_int = interleave_weights(w2_scale, n)
        
        # Compute activation scales
        w3_w1_pre_quant_max = torch.max(w1_pre_quant_scale, w3_pre_quant_scale)
        w3_w1_input_scale_max = input_scale.max()
        fc31_act_scale = (w3_w1_pre_quant_max / w3_w1_input_scale_max).to(config.output_dtype)
        fc2_act_scale = (w2_pre_quant_scale / input_scale).to(config.output_dtype).unsqueeze(-1)
        
        fc31_alpha = (weight_scale_2.squeeze(-1) * w3_w1_input_scale_max).float()
        fc2_alpha = (weight_scale_2.squeeze(-1) * input_scale.squeeze(-1)).float()
        
        zero_1 = torch.empty(0, dtype=config.output_dtype, device=device)
        zero_2 = torch.empty(0, dtype=config.output_dtype, device=device)
        
        # SM90 requires bfloat16 bit patterns for scales
        sm = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
        if sm >= 90:
            w3_w1_scales_out = w3_w1_scales_int.to(torch.bfloat16).view(config.output_dtype)
            w2_scales_out = w2_scales_int.to(torch.bfloat16).view(config.output_dtype)
            fc31_act_out = fc31_act_scale.to(torch.bfloat16).view(config.output_dtype)
            fc2_act_out = fc2_act_scale.to(torch.bfloat16).view(config.output_dtype)
        else:
            w3_w1_scales_out = w3_w1_scales_int.to(config.output_dtype)
            w2_scales_out = w2_scales_int.to(config.output_dtype)
            fc31_act_out = fc31_act_scale
            fc2_act_out = fc2_act_scale
        
        # Prepare quant_scales for W4A8
        quant_scales = (
            w3_w1_scales_out,   # fc1 weight scales
            w2_scales_out,      # fc2 weight scales
            fc31_act_out,       # fc1 activation scales
            fc2_act_out,        # fc2 activation scales
            zero_1,             # placeholder
            zero_2,             # placeholder
            fc31_alpha,         # fc1 alpha
            fc2_alpha,          # fc2 alpha
        )
        
        # Create routing
        router_logits = torch.randn(m, e, dtype=config.output_dtype, device=device)
        routing_weights, selected_experts = compute_routing(router_logits, top_k)
        
        output = torch.zeros_like(x)
        
        print(f"\n  Calling cutlass_fused_moe with Int4 weights × FP8 block-scale activations...")
        print(f"  Using Swiglu (gated) activation for W4A8 path")
        print(f"  Note: This may succeed on SM90 via the W4A8 path")
        print()
        
        # The key call - attempting Int4 × FP8 (W4A8) MoE with block-scale
        # NOTE: W4A8 uses Swiglu (default), NOT Relu2!
        _ = cutlass_fused_moe(
            x_fp8,  # FP8 input with block-scale quantization
            selected_experts.to(torch.int32),
            routing_weights,
            fc1_weights.view(torch.uint8),  # INT4 weights packed as uint8
            fc2_weights.view(torch.uint8),  # INT4 weights packed as uint8
            config.output_dtype,
            quant_scales=quant_scales,
            input_sf=x_scales_transposed,  # Block-scale factors [H/128, T]
            use_w4_group_scaling=True,  # Enable W4A8 path
            use_packed_weights=True,    # Weights are packed uint4x2
            use_deepseek_fp8_block_scale=True,  # Enable block-scale mode
            output=output,
            # activation_type=ActivationType.Swiglu,  # Default, gated activation
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: Int4 × FP8 (W4A8 + block-scale) completed!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        print(f"    Output sample: {output[0, :5]}")
        
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_frame = tb[-1]
            result["error_location"] = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        
        print(f"  ✗ ERROR CAPTURED:")
        print(f"    Type: {result['error_type']}")
        print(f"    Message: {result['error_message']}")
        print(f"    Location: {result['error_location']}")
        print(f"\n  Full Traceback:")
        print("  " + result["traceback"].replace("\n", "\n  "))
    
    return result


def test_nvfp4_x_bf16(config: MoEConfig) -> Dict[str, Any]:
    """
    Test NVFP4 (4-bit) weights × BF16 (16-bit) activations.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        Based on test_moe_nvfp4 skip condition, NVFP4 is only supported on 
        SM100, SM110, SM120 (Blackwell). This test should FAIL on SM90.
        
    NOTE: The user believes this should work on Hopper - this test will
    help surface whether that's the case or if NVFP4 requires Blackwell.
    
    NVFP4 uses:
    - fp4_quantize() for weight quantization
    - Weights passed as torch.long (int64) - 16 FP4 values per int64
    - Block scales in FP8 E4M3 format
    - Global scale factor
    """
    print_separator("TEST: NVFP4 × BF16 MoE")
    print_config(config)
    print()
    
    result = {
        "config": "NVFP4 × BF16",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        from flashinfer import fp4_quantize
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m = config.num_tokens
        k = config.hidden_size
        n = config.intermediate_size
        e = config.num_experts
        top_k = config.top_k
        quant_blocksize = 16
        
        round_up = lambda x, y: (x + y - 1) // y * y
        
        # For Relu2 (non-gated): w1 has shape [e, n, k]
        # For Swiglu (gated): w1 has shape [e, 2*n, k]
        # Let's use Swiglu for compatibility with standard NVFP4 path
        w1_n = 2 * n  # gated
        w1 = torch.randn((e, w1_n, k), device=device, dtype=torch.bfloat16) / 10
        w2 = torch.randn((e, k, n), device=device, dtype=torch.bfloat16) / 10
        
        # Create BF16 activations (NOT FP8)
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        
        print(f"  Created BF16 activations: shape={x.shape}, dtype={x.dtype}")
        
        # Quantize weights to NVFP4 using fp4_quantize
        FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
        FLOAT4_E2M1_MAX = 6.0
        
        sf_w1_n = round_up(w1_n, 128)
        sf_w1_k = round_up(k // quant_blocksize, 4)
        sf_w2_k = round_up(k, 128)
        sf_w2_n = round_up(n // quant_blocksize, 4)
        
        w1_q = torch.empty((e, w1_n, k // 2), device=device, dtype=torch.uint8)
        w2_q = torch.empty((e, k, n // 2), device=device, dtype=torch.uint8)
        w1_blockscale = torch.empty((e, sf_w1_n, sf_w1_k), device=device, dtype=torch.float8_e4m3fn)
        w2_blockscale = torch.empty((e, sf_w2_k, sf_w2_n), device=device, dtype=torch.float8_e4m3fn)
        w1_gs = torch.empty((e,), device=device, dtype=torch.float32)
        w2_gs = torch.empty((e,), device=device, dtype=torch.float32)
        
        print(f"  Quantizing weights to NVFP4 using fp4_quantize...")
        
        for expert in range(e):
            w1_amax = torch.abs(w1[expert]).max().to(torch.float32)
            w2_amax = torch.abs(w2[expert]).max().to(torch.float32)
            w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
            w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
            
            # This call may fail on SM90 if fp4_quantize requires Blackwell
            w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])
            w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])
        
        print(f"  NVFP4 weights created:")
        print(f"    w1_q shape={w1_q.shape}, dtype={w1_q.dtype}")
        print(f"    w1_blockscale shape={w1_blockscale.shape}, dtype={w1_blockscale.dtype}")
        
        # Create routing
        router_logits = torch.randn(m, e, dtype=torch.bfloat16, device=device)
        routing_weights, selected_experts = compute_routing(router_logits, top_k)
        
        # Prepare quant_scales for NVFP4 × BF16
        # Format: [fc1_act_global, fc1_weight_block, fc1_global, fc2_act_global, fc2_weight_block, fc2_global]
        a1_gs = torch.tensor(1.0, device=device, dtype=torch.float32)
        a2_gs = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        quant_scales = [
            a1_gs,                                    # fc1 activation global scale
            w1_blockscale.view(torch.int32),          # fc1 weight block scales
            1.0 / (a1_gs * w1_gs),                    # fc1 dequant scale
            a2_gs,                                    # fc2 activation global scale
            w2_blockscale.view(torch.int32),          # fc2 weight block scales
            1.0 / (a2_gs * w2_gs),                    # fc2 dequant scale
        ]
        
        output = torch.zeros_like(x)
        
        print(f"\n  Calling cutlass_fused_moe with NVFP4 weights × BF16 activations...")
        print(f"  Note: According to test_moe_nvfp4, NVFP4 is SM100+ only")
        print(f"  This test will reveal if NVFP4 × BF16 works on SM90")
        print()
        
        # The key call - attempting NVFP4 × BF16 MoE
        _ = cutlass_fused_moe(
            x,  # BF16 activations (NOT quantized)
            selected_experts.to(torch.int),
            routing_weights,
            w1_q.contiguous().view(torch.long),  # NVFP4 weights as int64
            w2_q.contiguous().view(torch.long),  # NVFP4 weights as int64
            config.output_dtype,
            quant_scales=quant_scales,
            output=output,
            activation_type=ActivationType.Swiglu,  # Gated for standard NVFP4 path
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: NVFP4 × BF16 completed!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        print(f"    Output sample: {output[0, :5]}")
        
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_frame = tb[-1]
            result["error_location"] = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        
        print(f"  ✗ ERROR CAPTURED:")
        print(f"    Type: {result['error_type']}")
        print(f"    Message: {result['error_message']}")
        print(f"    Location: {result['error_location']}")
        print(f"\n  Full Traceback:")
        print("  " + result["traceback"].replace("\n", "\n  "))
    
    return result


def test_mxfp4_x_bf16(config: MoEConfig) -> Dict[str, Any]:
    """
    Test MXFP4 (4-bit) weights × BF16 (16-bit) activations.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration SHOULD SUCCEED on SM90 according to test_moe_bf16_mxfp4
        which is explicitly marked as "BF16xMXFP4 is only supported on SM90".
        
    This uses:
    - uint8 packed weights (MXFP4)
    - uint8 scales (viewed as int32)
    - use_w4_group_scaling=True
    - BF16 activations (not quantized)
    """
    print_separator("TEST: MXFP4 × BF16 MoE (SM90 Supported)")
    print_config(config)
    print()
    
    result = {
        "config": "MXFP4 × BF16",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m = config.num_tokens
        k = config.hidden_size
        n = config.intermediate_size
        e = config.num_experts
        top_k = config.top_k
        
        # Create BF16 activations
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        
        print(f"  Created BF16 activations: shape={x.shape}, dtype={x.dtype}")
        
        # Create MXFP4 weights as uint8 (packed)
        # Shape: [e, 2*n, k//2] for gated (Swiglu), [e, n, k//2] for non-gated
        # Using gated shape for Swiglu
        w1 = torch.randint(0, 256, (e, 2 * n, k // 2), device=device, dtype=torch.uint8)
        w2 = torch.randint(0, 256, (e, k, n // 2), device=device, dtype=torch.uint8)
        
        # Scale factors: uint8 with shape matching group size (k//32)
        w1_scale = torch.randint(118, 123, (e, 2 * n, k // 32), device=device, dtype=torch.uint8)
        w2_scale = torch.randint(118, 123, (e, k, n // 32), device=device, dtype=torch.uint8)
        
        print(f"  Created MXFP4 weights:")
        print(f"    w1 shape={w1.shape}, dtype={w1.dtype}")
        print(f"    w1_scale shape={w1_scale.shape}, dtype={w1_scale.dtype}")
        print(f"    w2 shape={w2.shape}, dtype={w2.dtype}")
        
        # Create routing
        router_logits = torch.randn(m, e, dtype=torch.bfloat16, device=device)
        routing_weights, selected_experts = compute_routing(router_logits, top_k)
        
        output = torch.zeros_like(x)
        
        # Prepare quant_scales for BF16 × MXFP4 (only 2 scales needed)
        quant_scales = [
            w1_scale.view(torch.int32),
            w2_scale.view(torch.int32),
        ]
        
        print(f"\n  Calling cutlass_fused_moe with MXFP4 weights × BF16 activations...")
        print(f"  Using use_w4_group_scaling=True (SM90 path)")
        print(f"  This SHOULD work on SM90 according to test_moe_bf16_mxfp4")
        print()
        
        # The key call - attempting MXFP4 × BF16 MoE
        _ = cutlass_fused_moe(
            x,  # BF16 activations
            selected_experts.to(torch.int),
            routing_weights,
            w1.contiguous().view(torch.uint8),  # MXFP4 weights as uint8
            w2.contiguous().view(torch.uint8),  # MXFP4 weights as uint8
            torch.bfloat16,  # output dtype
            quant_scales=quant_scales,
            use_w4_group_scaling=True,  # Key flag for SM90 MXFP4 path
            output=output,
            # Default Swiglu activation (gated)
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: MXFP4 × BF16 completed!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        print(f"    Output sample: {output[0, :5]}")
        
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_frame = tb[-1]
            result["error_location"] = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        
        print(f"  ✗ ERROR CAPTURED:")
        print(f"    Type: {result['error_type']}")
        print(f"    Message: {result['error_message']}")
        print(f"    Location: {result['error_location']}")
        print(f"\n  Full Traceback:")
        print("  " + result["traceback"].replace("\n", "\n  "))
    
    return result


def test_int4_x_fp8_blockscale_relu2(config: MoEConfig) -> Dict[str, Any]:
    """
    Test Int4 (4-bit) weights × FP8 (8-bit) activations with Relu2 (non-gated) + block-scale.
    
    EXPECTED BEHAVIOR:
        This tests whether W4A8 supports non-gated (Relu2) activation with block-scale FP8.
        For non-gated MoE:
        - fc1 shape: [e, n, k//2] (NOT 2*n)
        - fc2 shape: [e, k, n//2]
        
    This may fail if W4A8 only supports gated activations (Swiglu).
    
    Block-scale FP8 quantization:
        - Activations are quantized with 128-element blocks
        - Each block has its own scale factor
    """
    print_separator("TEST: Int4 × FP8 MoE with Relu2 (Non-Gated + Block-Scale)")
    print_config(config)
    print()
    
    result = {
        "config": "Int4 × FP8 (W4A8) + Relu2 + Block-Scale",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m = config.num_tokens
        k = config.hidden_size
        n = config.intermediate_size
        e = config.num_experts
        top_k = config.top_k
        group_size = 128
        affine_coeff = 0.005
        
        # Create input and quantize to FP8 with BLOCK-SCALE
        x = torch.randn(m, k, dtype=config.output_dtype, device=device)
        x_fp8, x_scales = fp8_block_quant_1d(x, block_size=128)
        # Transpose scales for API compatibility: [T, H/128] -> [H/128, T]
        x_scales_transposed = x_scales.t().contiguous()
        
        print(f"  Created FP8 block-scale activations:")
        print(f"    x_fp8 shape: {x_fp8.shape}, dtype: {x_fp8.dtype}")
        print(f"    x_scales shape: {x_scales.shape} -> transposed: {x_scales_transposed.shape}")
        
        # =======================================================================
        # For Relu2 (NON-GATED), fc1 has shape [e, n, k//2] (not 2*n!)
        # Shape requirements for non-gated W4A8:
        #   - fc1: [num_experts, intermediate_size, hidden_size // 2]
        #   - fc2: [num_experts, hidden_size, intermediate_size // 2]
        # =======================================================================
        
        # Create INT4 quantized weights for NON-GATED MoE
        fc1_weights = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device=device)
        fc2_weights = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, device=device)
        
        print(f"  Created INT4 weights for NON-GATED (Relu2):")
        print(f"    fc1: shape={fc1_weights.shape} [e, n, k//2]")
        print(f"    fc2: shape={fc2_weights.shape} [e, k, n//2]")
        print(f"    Check: fc1.size(1)={fc1_weights.size(1)} == fc2.size(2)*2={fc2_weights.size(2)*2}")
        
        # Per-group weight scales (non-gated shape)
        fc1_scale = torch.randn(e, n, k // group_size, dtype=config.output_dtype, device=device) * affine_coeff
        fc2_scale = torch.randn(e, k, n // group_size, dtype=config.output_dtype, device=device) * affine_coeff
        
        # Per-channel pre-quant scales
        fc1_pre_quant_scale = torch.rand(e, k, dtype=config.output_dtype, device=device) * 0.1 + 0.95
        fc2_pre_quant_scale = torch.rand(e, n, dtype=config.output_dtype, device=device) * 0.1 + 0.95
        
        input_scale = torch.rand(e, 1, dtype=torch.float32, device=device) * 0.2 + 0.1
        weight_scale_2 = torch.ones(e, 1, dtype=torch.float32, device=device)
        
        # Interleave weights for TRTLLM format
        def interleave_weights(w: torch.Tensor, dim: int) -> torch.Tensor:
            interleave_factor = 4 if dim % 512 == 0 else (2 if dim % 256 == 0 else 1)
            s = w.shape
            w_interleaved = (
                w.reshape(s[0], s[1], s[2] // interleave_factor, interleave_factor)
                .permute(0, 2, 1, 3)
                .reshape(s[0], s[2] // interleave_factor, s[1] * interleave_factor)
                .contiguous()
            )
            return w_interleaved
        
        fc1_scales_int = interleave_weights(fc1_scale, k)
        fc2_scales_int = interleave_weights(fc2_scale, n)
        
        # Compute activation scales
        fc1_input_scale_max = input_scale.max()
        fc1_act_scale = (fc1_pre_quant_scale / fc1_input_scale_max).to(config.output_dtype)
        fc2_act_scale = (fc2_pre_quant_scale / input_scale).to(config.output_dtype).unsqueeze(-1)
        
        fc1_alpha = (weight_scale_2.squeeze(-1) * fc1_input_scale_max).float()
        fc2_alpha = (weight_scale_2.squeeze(-1) * input_scale.squeeze(-1)).float()
        
        zero_1 = torch.empty(0, dtype=config.output_dtype, device=device)
        zero_2 = torch.empty(0, dtype=config.output_dtype, device=device)
        
        # SM90 requires bfloat16 bit patterns for scales
        sm = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
        if sm >= 90:
            fc1_scales_out = fc1_scales_int.to(torch.bfloat16).view(config.output_dtype)
            fc2_scales_out = fc2_scales_int.to(torch.bfloat16).view(config.output_dtype)
            fc1_act_out = fc1_act_scale.to(torch.bfloat16).view(config.output_dtype)
            fc2_act_out = fc2_act_scale.to(torch.bfloat16).view(config.output_dtype)
        else:
            fc1_scales_out = fc1_scales_int.to(config.output_dtype)
            fc2_scales_out = fc2_scales_int.to(config.output_dtype)
            fc1_act_out = fc1_act_scale
            fc2_act_out = fc2_act_scale
        
        # Prepare quant_scales for W4A8
        quant_scales = (
            fc1_scales_out,     # fc1 weight scales
            fc2_scales_out,     # fc2 weight scales
            fc1_act_out,        # fc1 activation scales
            fc2_act_out,        # fc2 activation scales
            zero_1,             # placeholder
            zero_2,             # placeholder
            fc1_alpha,          # fc1 alpha
            fc2_alpha,          # fc2 alpha
        )
        
        # Create routing
        router_logits = torch.randn(m, e, dtype=config.output_dtype, device=device)
        routing_weights, selected_experts = compute_routing(router_logits, top_k)
        
        output = torch.zeros_like(x)
        
        print(f"\n  Calling cutlass_fused_moe with Int4 weights × FP8 block-scale + Relu2 (non-gated)...")
        print(f"  This tests if W4A8 supports non-gated activation with block-scale FP8")
        print()
        
        # The key call - attempting Int4 × FP8 (block-scale) with Relu2
        _ = cutlass_fused_moe(
            x_fp8,  # FP8 with block-scale quantization
            selected_experts.to(torch.int32),
            routing_weights,
            fc1_weights.view(torch.uint8),
            fc2_weights.view(torch.uint8),
            config.output_dtype,
            quant_scales=quant_scales,
            input_sf=x_scales_transposed,  # Block-scale factors [H/128, T]
            use_w4_group_scaling=True,
            use_packed_weights=True,
            use_deepseek_fp8_block_scale=True,  # Enable block-scale mode
            output=output,
            activation_type=ActivationType.Relu2,  # NON-GATED!
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: Int4 × FP8 (block-scale) + Relu2 completed!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        print(f"    Output sample: {output[0, :5]}")
        
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_frame = tb[-1]
            result["error_location"] = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        
        print(f"  ✗ ERROR CAPTURED:")
        print(f"    Type: {result['error_type']}")
        print(f"    Message: {result['error_message']}")
        print(f"    Location: {result['error_location']}")
        print(f"\n  Full Traceback:")
        print("  " + result["traceback"].replace("\n", "\n  "))
    
    return result


def test_fp8_x_fp8_blockscale(config: MoEConfig) -> Dict[str, Any]:
    """
    Test FP8 (8-bit) weights × FP8 (8-bit) activations with DeepSeek block-scale.
    
    This is the dedicated DeepSeek-style W8A8 configuration using block-scale
    quantization for both weights and activations.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration SHOULD SUCCEED on SM90 because it uses the dedicated
        trtllm_fp8_block_scale_moe API designed for DeepSeek-style quantization.
        
    Block-scale FP8 quantization:
        - 128x128 blocks for weights
        - Per-token × 128-col blocks for activations
        - Scale shapes:
            - hidden_states_scale: [H/128, T] (transposed)
            - gemm1_weights_scale: [E, 2*I/128, H/128]
            - gemm2_weights_scale: [E, H/128, I/128]
    """
    print_separator("TEST: FP8 × FP8 MoE (DeepSeek Block-Scale)")
    print(f"  Weight Format:      FP8 (E4M3) with 128x128 Block-Scale")
    print(f"  Activation Format:  FP8 (E4M3) with DeepSeek Block-Scale (128)")
    print(f"  Output dtype:       {config.output_dtype}")
    print(f"  Num Tokens:         {config.num_tokens}")
    print(f"  Hidden Size:        {config.hidden_size}")
    print(f"  Intermediate Size:  {config.intermediate_size}")
    print(f"  Num Experts:        {config.num_experts}")
    print(f"  Top-K:              {config.top_k}")
    print(f"  Activation Type:    Swiglu (gated)")
    print()
    
    result = {
        "config": "FP8 × FP8 (DeepSeek Block-Scale)",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m = config.num_tokens
        k = config.hidden_size
        n = config.intermediate_size
        e = config.num_experts
        top_k = config.top_k
        
        # DeepSeek-style routing parameters
        n_group = 1  # simplified for testing
        topk_group = 1
        routed_scaling_factor = 1.0
        
        # Create BF16 activations and quantize to FP8 with block-scale
        x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        x_fp8, x_scales = fp8_block_quant_1d(x_bf16, block_size=128)
        # Transpose scales for API: [T, H/128] -> [H/128, T]
        x_scales_transposed = x_scales.t().contiguous()
        
        print(f"  Created FP8 block-scale activations:")
        print(f"    x_fp8 shape: {x_fp8.shape}, dtype: {x_fp8.dtype}")
        print(f"    x_scales_transposed shape: {x_scales_transposed.shape}")
        
        # Create BF16 weights and quantize to FP8 with 2D block-scale
        # For Swiglu (gated): w1 has shape [e, 2*n, k]
        w1_bf16 = torch.randn((e, 2 * n, k), device=device, dtype=torch.bfloat16) / 10
        w2_bf16 = torch.randn((e, k, n), device=device, dtype=torch.bfloat16) / 10
        
        w1_fp8, w1_scales = fp8_block_quant_2d(w1_bf16, block_size=128)
        w2_fp8, w2_scales = fp8_block_quant_2d(w2_bf16, block_size=128)
        
        print(f"  Created FP8 block-scale weights:")
        print(f"    w1_fp8 shape: {w1_fp8.shape}, w1_scales shape: {w1_scales.shape}")
        print(f"    w2_fp8 shape: {w2_fp8.shape}, w2_scales shape: {w2_scales.shape}")
        
        # Create routing logits and bias
        routing_logits = torch.randn(m, e, dtype=torch.float32, device=device)
        routing_bias = torch.zeros(e, dtype=torch.bfloat16, device=device)
        
        print(f"\n  Calling trtllm_fp8_block_scale_moe (DeepSeek-style API)...")
        print(f"  This SHOULD work on SM90 (Hopper)")
        print()
        
        # The key call - DeepSeek-style FP8 block-scale MoE
        output = trtllm_fp8_block_scale_moe(
            routing_logits,
            routing_bias,
            x_fp8,
            x_scales_transposed,
            w1_fp8,
            w1_scales.to(torch.float32),
            w2_fp8,
            w2_scales.to(torch.float32),
            e,  # num_experts
            top_k,
            n_group,
            topk_group,
            n,  # intermediate_size
            0,  # local_expert_offset
            e,  # local_num_experts
            routed_scaling_factor,
            routing_method_type=0,  # Default routing
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: FP8 × FP8 (DeepSeek block-scale) completed!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        print(f"    Output sample: {output[0, :5]}")
        
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_frame = tb[-1]
            result["error_location"] = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        
        print(f"  ✗ ERROR CAPTURED:")
        print(f"    Type: {result['error_type']}")
        print(f"    Message: {result['error_message']}")
        print(f"    Location: {result['error_location']}")
        print(f"\n  Full Traceback:")
        print("  " + result["traceback"].replace("\n", "\n  "))
    
    return result


# ==============================================================================
# Main Test Driver
# ==============================================================================

def main():
    """Run all mixed-precision 4-bit × 8-bit MoE tests with block-scale FP8."""
    print_separator("FlashInfer Mixed Precision 4-bit × 8-bit MoE Error-Surfacing")
    print("    (DeepSeek Block-Scale FP8 Quantization Version)")
    
    # Print device info
    device_info = get_device_info()
    print("\nDEVICE INFORMATION:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    if device_info.get("error"):
        print("\n❌ CUDA not available. Exiting.")
        sys.exit(1)
    
    sm_version = device_info.get("sm_version", 0)
    print(f"\n  Target SM: {sm_version}")
    if sm_version >= 90 and sm_version < 100:
        print("  Architecture: Hopper (SM90)")
        print("  Expected: NVFP4/MXFP4 × FP8 should FAIL")
        print("  Expected: Int4 × FP8 (W4A8) may SUCCEED")
        print("  Expected: FP8 × FP8 (DeepSeek block-scale) SHOULD SUCCEED")
    elif sm_version >= 100:
        print("  Architecture: Blackwell (SM100+)")
        print("  Expected: All configurations may SUCCEED")
    else:
        print(f"  Architecture: Pre-Hopper (SM{sm_version})")
        print("  Expected: All configurations may FAIL")
    
    print("\n  NOTE: This version uses DeepSeek-style BLOCK-SCALE FP8 quantization")
    print("        with 128-element blocks for activation quantization.")
    
    # Test configuration
    config = MoEConfig(
        num_tokens=4,
        hidden_size=128,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        output_dtype=torch.bfloat16,
    )
    
    results = []
    
    # Run all tests
    # ==========================================================================
    # 4-bit weights × 8-bit (FP8 block-scale) activations
    # ==========================================================================
    
    # Test 1: NVFP4 × FP8 (block-scale)
    config.weight_format = WeightFormat.NVFP4
    results.append(test_nvfp4_x_fp8_blockscale(config))
    
    # Test 2: MXFP4 × FP8 (block-scale)
    config.weight_format = WeightFormat.MXFP4
    results.append(test_mxfp4_x_fp8_blockscale(config))
    
    # Test 3: Int4 × FP8 with Swiglu (gated) - standard W4A8 path with block-scale
    config.weight_format = WeightFormat.INT4
    results.append(test_int4_x_fp8_blockscale(config))
    
    # Test 4: Int4 × FP8 with Relu2 (non-gated) - tests non-gated W4A8 support with block-scale
    results.append(test_int4_x_fp8_blockscale_relu2(config))
    
    # ==========================================================================
    # 4-bit weights × 16-bit (BF16) activations - should work on SM90
    # ==========================================================================
    
    # Test 5: NVFP4 × BF16 (may require SM100+, testing to verify)
    results.append(test_nvfp4_x_bf16(config))
    
    # Test 6: MXFP4 × BF16 (explicitly supported on SM90)
    results.append(test_mxfp4_x_bf16(config))
    
    # ==========================================================================
    # 8-bit weights × 8-bit (FP8 block-scale) activations - DeepSeek W8A8
    # ==========================================================================
    
    # Test 7: FP8 × FP8 (DeepSeek block-scale) - should work on SM90
    results.append(test_fp8_x_fp8_blockscale(config))
    
    # Summary
    print_separator("SUMMARY")
    
    print("TEST RESULTS:")
    for r in results:
        status = "✓ SUCCESS" if r["success"] else "✗ FAILED"
        print(f"\n  {r['config']}: {status}")
        if not r["success"]:
            print(f"    Error Type: {r['error_type']}")
            print(f"    Error Message: {r['error_message'][:100]}..." if len(str(r['error_message'])) > 100 else f"    Error Message: {r['error_message']}")
    
    print("\n" + "="*80)
    print("  Analysis: Failure Modes")
    print("="*80)
    
    for r in results:
        if not r["success"]:
            print(f"\n  [{r['config']}]")
            print(f"  Error occurred at: Python validation / Kernel selection / Runtime")
            
            # Analyze error type
            if "NotImplementedError" in str(r.get("error_type", "")):
                print("  Failure Mode: Kernel not implemented for this architecture")
            elif "RuntimeError" in str(r.get("error_type", "")):
                print("  Failure Mode: Runtime kernel launch failure")
            elif "ValueError" in str(r.get("error_type", "")):
                print("  Failure Mode: Python-level validation rejection")
            else:
                print(f"  Failure Mode: {r.get('error_type', 'Unknown')}")
    
    print("\n" + "="*80)
    print("  Script completed. See above for detailed error traces.")
    print("="*80 + "\n")
    
    # Return non-zero if any test that was expected to fail actually succeeded unexpectedly
    # For SM90, NVFP4 and MXFP4 should fail
    if sm_version >= 90 and sm_version < 100:
        unexpected_successes = [r for r in results if r["success"] and r["config"] in ["NVFP4 × FP8 (Block-Scale)", "MXFP4 × FP8 (Block-Scale)"]]
        if unexpected_successes:
            print("⚠️  Unexpected successes on SM90 - this may indicate new kernel support!")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

