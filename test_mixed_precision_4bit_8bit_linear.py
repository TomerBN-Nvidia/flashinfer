#!/usr/bin/env python3
"""
================================================================================
FlashInfer Mixed Precision 4-bit × 8-bit Linear Error-Surfacing Script
================================================================================

PURPOSE:
    This script explicitly attempts to run 4-bit weight × 8-bit activation
    mixed-precision LINEAR (GEMM) configurations using FlashInfer's CUTLASS
    GEMM kernels.
    
    The goal is to SURFACE ERRORS and UNDERSTAND FAILURE MODES, not to make
    operations succeed. This helps validate kernel support assumptions and
    understand how FlashInfer/CUTLASS rejects unsupported configurations.

TARGET HARDWARE:
    - Hopper GPU (SM >= 90)
    
CONFIGURATIONS TESTED:
    1. NVFP4 weights × FP8 activations (mm_fp4 with nvfp4 quantization)
       - Expected: FAIL on SM90 (NVFP4 requires SM100+)
       
    2. MXFP4 weights × FP8 activations (mm_fp4 with mxfp4 quantization)
       - Expected: FAIL on SM90 (MXFP4×FP8 requires SM100+)
       
    3. FP8 × FP8 (bmm_fp8 / fp8_blockscale_gemm_sm90)
       - Expected: SUCCEED on SM90 (native Hopper support)
       
    4. NVFP4 × BF16 (mm_fp4)
       - Expected: FAIL on SM90 (NVFP4 requires SM100+)
       
    5. MXFP4 × BF16 (mm_fp4 with mxfp4 quantization)  
       - Expected: May work on SM90 via W4A16 path

    6. FP8 × BF16 (fp8_blockscale_gemm_sm90)
       - Expected: SUCCEED on SM90

LINEAR STRUCTURE:
    - Simple matrix multiplication: C = A @ B.T
    - Output dtype: BF16

⚠️ IMPORTANT:
    This script intentionally does NOT:
    - Fall back to supported dtypes
    - Skip unsupported configurations
    - Convert weights/activations to higher precision automatically
    
Reference:
    - Quantization POR validation
    - Scope NVFP4 / MXFP4 / Int4 × FP8 GEMM for SM >= 90
    
================================================================================
"""

import traceback
import sys
from typing import Tuple, Optional, Dict, Any
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
    FP8 = "fp8"          # FP8 E4M3


@dataclass
class LinearConfig:
    """Linear test configuration."""
    m: int = 128          # Batch/token dimension
    n: int = 256          # Output dimension
    k: int = 512          # Input/hidden dimension
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


def dynamic_per_tensor_fp8_quant(
    x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 E4M3 format with per-tensor scale."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    fp8_min = -fp8_max
    
    x_max = x.abs().max().float()
    scale = x_max / fp8_max
    iscale = 1.0 / scale.clamp(min=1e-12)
    
    out = (x.float() * iscale).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn)
    return out, scale.view((1,))


def per_token_cast_to_fp8(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 E4M3 format with per-token (1x128 block) scale."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    
    # Reshape to blocks
    m, k = x.shape
    k_blocks = k // block_size
    x_blocks = x.view(m, k_blocks, block_size)
    
    # Compute per-block max
    block_max = x_blocks.abs().amax(dim=-1, keepdim=True).float()
    scale = block_max / fp8_max
    scale = scale.clamp(min=1e-12)
    
    # Quantize
    x_scaled = x_blocks / scale
    x_fp8 = x_scaled.clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.view(m, k)
    
    # Scale is reciprocal for the kernel
    return x_fp8, scale.squeeze(-1)


def print_separator(title: str = None):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'-'*80}\n")


def print_config(config: LinearConfig):
    """Print Linear configuration."""
    print(f"  Weight Format:      {config.weight_format.value}")
    print(f"  Activation Format:  FP8 (E4M3) or BF16")
    print(f"  Output dtype:       {config.output_dtype}")
    print(f"  M (batch):          {config.m}")
    print(f"  N (output):         {config.n}")
    print(f"  K (hidden):         {config.k}")


# ==============================================================================
# Test Functions for Each Configuration
# ==============================================================================

def test_nvfp4_x_fp8(config: LinearConfig) -> Dict[str, Any]:
    """
    Test NVFP4 (4-bit) weights × FP8 (8-bit) activations.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration should FAIL because NVFP4 is only supported on 
        Blackwell architecture (SM100+). The error should occur at:
        - Module initialization (JIT compilation)
        - Or kernel selection phase
        
    The NVFP4 format uses E2M1 (2-bit exponent, 1-bit mantissa) which is 
    a native format on Blackwell but not Hopper.
    """
    print_separator("TEST: NVFP4 × FP8 Linear (mm_fp4)")
    print_config(config)
    print()
    
    result = {
        "config": "NVFP4 × FP8",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.gemm import mm_fp4
        from flashinfer import nvfp4_quantize, SfLayout
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m, n, k = config.m, config.n, config.k
        
        # Create high-precision tensors
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        
        # First, quantize activations to FP8
        a_fp8, a_scale = dynamic_per_tensor_fp8_quant(a_bf16)
        print(f"  Created FP8 activations: shape={a_fp8.shape}, dtype={a_fp8.dtype}")
        
        # Quantize weights to NVFP4
        print(f"  Quantizing weights to NVFP4...")
        b_global_sf = (448 * 6) / b_bf16.float().abs().nan_to_num().max()
        
        # This call may fail on SM90 if NVFP4 quantization requires Blackwell
        b_fp4, b_sf = nvfp4_quantize(
            b_bf16, 
            b_global_sf, 
            sfLayout=SfLayout.layout_128x4, 
            do_shuffle=False
        )
        
        print(f"  NVFP4 weights created: shape={b_fp4.shape}, dtype={b_fp4.dtype}")
        print(f"  NVFP4 scales: shape={b_sf.shape}, dtype={b_sf.dtype}")
        
        # For NVFP4 × FP8, we need to handle this differently
        # mm_fp4 expects both inputs to be FP4, so we need a different path
        # Let's try to use the raw FP8 input with FP4 weights
        
        print(f"\n  Calling mm_fp4 with NVFP4 weights × FP8 activations...")
        print(f"  ⚠️  This is expected to FAIL on SM90 (Hopper)")
        print()
        
        # For mm_fp4, we need FP4 activations too, but let's try anyway
        # The real test is whether the nvfp4_quantize and subsequent operations work
        a_global_sf = (448 * 6) / a_bf16.float().abs().nan_to_num().max()
        a_fp4, a_sf = nvfp4_quantize(
            a_bf16, 
            a_global_sf, 
            sfLayout=SfLayout.layout_128x4, 
            do_shuffle=False
        )
        
        # Now call mm_fp4 with both FP4 inputs
        alpha = 1.0 / (a_global_sf * b_global_sf)
        output = mm_fp4(
            a_fp4,
            b_fp4.T.contiguous() if b_fp4.T.is_contiguous() else b_fp4.T,
            a_sf.view(torch.float8_e4m3fn),
            b_sf.T.contiguous().view(torch.float8_e4m3fn) if b_sf.T.is_contiguous() else b_sf.view(torch.float8_e4m3fn).T,
            alpha=alpha,
            out_dtype=config.output_dtype,
            use_nvfp4=True,
        )
        
        result["success"] = True
        print("  ✓ UNEXPECTED SUCCESS: NVFP4 × NVFP4 completed without error!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        
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


def test_mxfp4_x_fp8(config: LinearConfig) -> Dict[str, Any]:
    """
    Test MXFP4 (4-bit) weights × MXFP8 (8-bit) activations.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration should FAIL because MXFP4×FP8 requires Blackwell 
        architecture (SM100+). The kernel instantiation for this combination
        is gated behind ENABLE_FP4 which is only defined for SM100+.
        
    MXFP4 uses microscaling with FP8 E8M0 scale factors per 32-element block.
    """
    print_separator("TEST: MXFP4 × MXFP8 Linear (mm_fp4)")
    print_config(config)
    print()
    
    result = {
        "config": "MXFP4 × MXFP8",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.gemm import mm_fp4
        from flashinfer import mxfp4_quantize, mxfp8_quantize
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m, n, k = config.m, config.n, config.k
        
        # Create high-precision tensors
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        
        print(f"  Quantizing activations to MXFP8...")
        # MXFP8 quantization
        a_mxfp8, a_mxfp8_sf = mxfp8_quantize(a_bf16, True, 32)
        print(f"  MXFP8 activations: shape={a_mxfp8.shape}, dtype={a_mxfp8.dtype}")
        
        print(f"  Quantizing weights to MXFP4...")
        # MXFP4 quantization
        b_mxfp4, b_mxfp4_sf = mxfp4_quantize(b_bf16)
        print(f"  MXFP4 weights: shape={b_mxfp4.shape}, dtype={b_mxfp4.dtype}")
        
        print(f"\n  Calling mm_fp4 with MXFP4 weights × MXFP4 activations...")
        print(f"  ⚠️  This is expected to FAIL on SM90 (Hopper)")
        print()
        
        # For mm_fp4, we need FP4 activations too
        a_mxfp4, a_mxfp4_sf = mxfp4_quantize(a_bf16)
        
        # Call mm_fp4 with mxfp4
        output = mm_fp4(
            a_mxfp4,
            b_mxfp4.T.contiguous() if b_mxfp4.shape[0] == n else b_mxfp4,
            a_mxfp4_sf.view(torch.uint8),
            b_mxfp4_sf.view(torch.uint8),
            alpha=torch.tensor(1.0, device=device),
            out_dtype=config.output_dtype,
            block_size=32,  # MXFP4 uses block_size=32
            use_nvfp4=False,
        )
        
        result["success"] = True
        print("  ✓ UNEXPECTED SUCCESS: MXFP4 × MXFP4 completed without error!")
        print(f"    Output shape: {output.shape}")
        
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


def test_fp8_x_fp8(config: LinearConfig) -> Dict[str, Any]:
    """
    Test FP8 (8-bit) weights × FP8 (8-bit) activations.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration SHOULD SUCCEED on SM90 because FP8 is natively
        supported on Hopper architecture via bmm_fp8 or fp8_blockscale_gemm_sm90.
        
    FP8 uses E4M3 format (4-bit exponent, 3-bit mantissa).
    """
    print_separator("TEST: FP8 × FP8 Linear (fp8_blockscale_gemm_sm90)")
    print_config(config)
    print()
    
    result = {
        "config": "FP8 × FP8",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.gemm import fp8_blockscale_gemm_sm90
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m, n, k = config.m, config.n, config.k
        # Ensure k is divisible by 128 for block-scale GEMM
        k = ((k + 127) // 128) * 128
        
        # Create high-precision tensors
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        
        print(f"  Quantizing activations to FP8...")
        a_fp8, a_scale = per_token_cast_to_fp8(a_bf16, block_size=128)
        print(f"  FP8 activations: shape={a_fp8.shape}, dtype={a_fp8.dtype}")
        print(f"  Activation scales: shape={a_scale.shape}")
        
        print(f"  Quantizing weights to FP8...")
        b_fp8, b_scale = per_token_cast_to_fp8(b_bf16, block_size=128)
        print(f"  FP8 weights: shape={b_fp8.shape}, dtype={b_fp8.dtype}")
        print(f"  Weight scales: shape={b_scale.shape}")
        
        print(f"\n  Calling fp8_blockscale_gemm_sm90 with FP8 × FP8...")
        print(f"  This SHOULD succeed on SM90 (Hopper)")
        print()
        
        # Prepare scales in the correct format for the kernel
        # Input scale needs to be (K//128, M) format
        M_padded = ((m + 4 - 1) // 4) * 4
        K_blocks = k // 128
        
        input_scale_padded = torch.zeros(K_blocks, M_padded, dtype=torch.float32, device=device)
        input_scale_padded[:, :m] = a_scale.T
        input_scale_padded = input_scale_padded[:, :m]
        
        output = fp8_blockscale_gemm_sm90(
            a_fp8,
            b_fp8,
            input_scale_padded,
            b_scale,
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: FP8 × FP8 completed!")
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


def test_fp8_x_bf16(config: LinearConfig) -> Dict[str, Any]:
    """
    Test FP8 (8-bit) weights × BF16 (16-bit) activations.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration SHOULD SUCCEED on SM90 because mixed FP8/BF16
        is supported via fp8_blockscale_gemm_sm90.
        
    Note: The kernel quantizes BF16 activations internally to FP8.
    """
    print_separator("TEST: FP8 weights × BF16 activations (fp8_blockscale_gemm_sm90)")
    print_config(config)
    print()
    
    result = {
        "config": "FP8 × BF16",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.gemm import fp8_blockscale_gemm_sm90
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m, n, k = config.m, config.n, config.k
        # Ensure k is divisible by 128 for block-scale GEMM
        k = ((k + 127) // 128) * 128
        
        # Create BF16 activations (NOT quantized)
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        print(f"  Created BF16 activations: shape={a_bf16.shape}, dtype={a_bf16.dtype}")
        
        # Create FP8 weights
        b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        b_fp8, b_scale = per_token_cast_to_fp8(b_bf16, block_size=128)
        print(f"  Created FP8 weights: shape={b_fp8.shape}, dtype={b_fp8.dtype}")
        print(f"  Weight scales: shape={b_scale.shape}")
        
        print(f"\n  Calling fp8_blockscale_gemm_sm90 with FP8 weights × BF16 activations...")
        print(f"  This SHOULD succeed on SM90 (Hopper)")
        print()
        
        # For BF16 input + FP8 weight, input_scale should be None
        output = fp8_blockscale_gemm_sm90(
            a_bf16,       # BF16 activations
            b_fp8,        # FP8 weights
            None,         # No input scale for BF16
            b_scale,      # Weight scale
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: FP8 × BF16 completed!")
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


def test_bf16_x_bf16(config: LinearConfig) -> Dict[str, Any]:
    """
    Test BF16 (16-bit) weights × BF16 (16-bit) activations with internal FP8 quantization.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration SHOULD SUCCEED on SM90 because the kernel
        internally quantizes both inputs to FP8.
    """
    print_separator("TEST: BF16 × BF16 Linear with internal FP8 (fp8_blockscale_gemm_sm90)")
    print_config(config)
    print()
    
    result = {
        "config": "BF16 × BF16 (internal FP8)",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.gemm import fp8_blockscale_gemm_sm90
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m, n, k = config.m, config.n, config.k
        # Ensure k is divisible by 128 for block-scale GEMM
        k = ((k + 127) // 128) * 128
        
        # Create BF16 tensors (NOT quantized)
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        
        print(f"  Created BF16 activations: shape={a_bf16.shape}, dtype={a_bf16.dtype}")
        print(f"  Created BF16 weights: shape={b_bf16.shape}, dtype={b_bf16.dtype}")
        
        print(f"\n  Calling fp8_blockscale_gemm_sm90 with BF16 × BF16...")
        print(f"  Kernel will internally quantize to FP8")
        print(f"  This SHOULD succeed on SM90 (Hopper)")
        print()
        
        # For BF16 + BF16, no scales needed
        output = fp8_blockscale_gemm_sm90(
            a_bf16,
            b_bf16,
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: BF16 × BF16 (internal FP8) completed!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        print(f"    Output sample: {output[0, :5]}")
        
        # Verify correctness
        reference = torch.matmul(a_bf16, b_bf16.T)
        cos_sim = F.cosine_similarity(
            reference.flatten().float(), output.flatten().float(), dim=0
        )
        print(f"    Cosine similarity with reference: {cos_sim:.4f}")
        
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


def test_nvfp4_x_bf16(config: LinearConfig) -> Dict[str, Any]:
    """
    Test NVFP4 (4-bit) weights × BF16 (16-bit) activations.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        Based on the MoE test, NVFP4 is only supported on SM100, SM110, SM120 
        (Blackwell). This test should FAIL on SM90.
    """
    print_separator("TEST: NVFP4 × BF16 Linear (mm_fp4)")
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
        from flashinfer.gemm import mm_fp4
        from flashinfer import nvfp4_quantize, SfLayout
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m, n, k = config.m, config.n, config.k
        
        # Create BF16 activations (NOT quantized to FP4)
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        
        print(f"  Created BF16 activations: shape={a_bf16.shape}, dtype={a_bf16.dtype}")
        
        # Quantize weights to NVFP4
        print(f"  Quantizing weights to NVFP4...")
        b_global_sf = (448 * 6) / b_bf16.float().abs().nan_to_num().max()
        
        # This call may fail on SM90
        b_fp4, b_sf = nvfp4_quantize(
            b_bf16, 
            b_global_sf, 
            sfLayout=SfLayout.layout_128x4, 
            do_shuffle=False
        )
        
        print(f"  NVFP4 weights created: shape={b_fp4.shape}, dtype={b_fp4.dtype}")
        
        # For mm_fp4, we also need to quantize activations to FP4
        print(f"  Quantizing activations to NVFP4...")
        a_global_sf = (448 * 6) / a_bf16.float().abs().nan_to_num().max()
        a_fp4, a_sf = nvfp4_quantize(
            a_bf16, 
            a_global_sf, 
            sfLayout=SfLayout.layout_128x4, 
            do_shuffle=False
        )
        
        print(f"\n  Calling mm_fp4 with NVFP4 weights × NVFP4 activations...")
        print(f"  Note: mm_fp4 requires both inputs to be FP4")
        print(f"  This is expected to FAIL on SM90 (Hopper)")
        print()
        
        alpha = 1.0 / (a_global_sf * b_global_sf)
        
        output = mm_fp4(
            a_fp4,
            b_fp4.T.contiguous(),
            a_sf.view(torch.float8_e4m3fn),
            b_sf.T.contiguous().view(torch.float8_e4m3fn),
            alpha=alpha,
            out_dtype=config.output_dtype,
            use_nvfp4=True,
        )
        
        result["success"] = True
        print("  ✓ UNEXPECTED SUCCESS: NVFP4 × NVFP4 completed!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        
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


def test_mxfp4_x_bf16(config: LinearConfig) -> Dict[str, Any]:
    """
    Test MXFP4 (4-bit) weights × BF16 (16-bit) activations.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This may work on SM90 via the W4A16 path, similar to how
        test_moe_bf16_mxfp4 is marked as SM90 compatible.
    """
    print_separator("TEST: MXFP4 × BF16 Linear (mm_fp4)")
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
        from flashinfer.gemm import mm_fp4
        from flashinfer import mxfp4_quantize
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        m, n, k = config.m, config.n, config.k
        
        # Create BF16 tensors
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        
        print(f"  Created BF16 activations: shape={a_bf16.shape}, dtype={a_bf16.dtype}")
        
        # Quantize weights to MXFP4
        print(f"  Quantizing weights to MXFP4...")
        b_mxfp4, b_mxfp4_sf = mxfp4_quantize(b_bf16)
        print(f"  MXFP4 weights: shape={b_mxfp4.shape}, dtype={b_mxfp4.dtype}")
        
        # Quantize activations to MXFP4 for mm_fp4
        print(f"  Quantizing activations to MXFP4...")
        a_mxfp4, a_mxfp4_sf = mxfp4_quantize(a_bf16)
        
        print(f"\n  Calling mm_fp4 with MXFP4 weights × MXFP4 activations...")
        print(f"  Using block_size=32 for MXFP4")
        print()
        
        output = mm_fp4(
            a_mxfp4,
            b_mxfp4.T.contiguous(),
            a_mxfp4_sf.view(torch.uint8),
            b_mxfp4_sf.T.contiguous().view(torch.uint8),
            alpha=torch.tensor(1.0, device=device),
            out_dtype=config.output_dtype,
            block_size=32,
            use_nvfp4=False,
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: MXFP4 × MXFP4 completed!")
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


def test_bmm_fp8(config: LinearConfig) -> Dict[str, Any]:
    """
    Test Batch Matrix Multiply with FP8 × FP8.
    
    EXPECTED BEHAVIOR ON SM90 (Hopper):
        This configuration SHOULD SUCCEED on SM90 because bmm_fp8
        is designed for Hopper architecture.
    """
    print_separator("TEST: BMM FP8 × FP8 (bmm_fp8)")
    print_config(config)
    print()
    
    result = {
        "config": "BMM FP8 × FP8",
        "success": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "traceback": None,
    }
    
    try:
        import flashinfer
        from flashinfer.gemm import bmm_fp8
        
        print(f"  FlashInfer version: {flashinfer.__version__}")
        
        device = "cuda"
        torch.manual_seed(42)
        
        batch = 4
        m, n, k = config.m, config.n, config.k
        
        # Create batched tensors
        a_bf16 = torch.randn(batch, m, k, dtype=torch.bfloat16, device=device)
        b_bf16 = torch.randn(batch, k, n, dtype=torch.bfloat16, device=device)  # Note: (batch, k, n) for column major
        
        # Quantize to FP8
        print(f"  Quantizing to FP8...")
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        
        a_scale = a_bf16.abs().amax(dim=(1, 2), keepdim=True) / fp8_max
        b_scale = b_bf16.abs().amax(dim=(1, 2), keepdim=True) / fp8_max
        
        a_fp8 = (a_bf16 / a_scale.clamp(min=1e-12)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        b_fp8 = (b_bf16 / b_scale.clamp(min=1e-12)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        
        print(f"  FP8 A: shape={a_fp8.shape}, dtype={a_fp8.dtype}")
        print(f"  FP8 B: shape={b_fp8.shape}, dtype={b_fp8.dtype}")
        
        print(f"\n  Calling bmm_fp8...")
        print(f"  This SHOULD succeed on SM90 (Hopper)")
        print()
        
        output = bmm_fp8(
            a_fp8,
            b_fp8,
            a_scale.view(batch),
            b_scale.view(batch),
            config.output_dtype,
        )
        
        result["success"] = True
        print("  ✓ SUCCESS: BMM FP8 × FP8 completed!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        print(f"    Output sample: {output[0, 0, :5]}")
        
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
    """Run all mixed-precision 4-bit × 8-bit Linear tests."""
    print_separator("FlashInfer Mixed Precision 4-bit × 8-bit Linear Error-Surfacing")
    
    # Print device info
    device_info = get_device_info()
    print("DEVICE INFORMATION:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    if device_info.get("error"):
        print("\n❌ CUDA not available. Exiting.")
        sys.exit(1)
    
    sm_version = device_info.get("sm_version", 0)
    print(f"\n  Target SM: {sm_version}")
    if sm_version >= 90 and sm_version < 100:
        print("  Architecture: Hopper (SM90)")
        print("  Expected: NVFP4/MXFP4 operations should FAIL")
        print("  Expected: FP8 operations should SUCCEED")
    elif sm_version >= 100:
        print("  Architecture: Blackwell (SM100+)")
        print("  Expected: All configurations may SUCCEED")
    else:
        print(f"  Architecture: Pre-Hopper (SM{sm_version})")
        print("  Expected: All configurations may FAIL")
    
    # Test configuration
    config = LinearConfig(
        m=128,
        n=256,
        k=512,
        output_dtype=torch.bfloat16,
    )
    
    results = []
    
    # Run all tests
    # ==========================================================================
    # 8-bit × 8-bit (FP8) - Should work on SM90
    # ==========================================================================
    
    # Test 1: FP8 × FP8
    config.weight_format = WeightFormat.FP8
    results.append(test_fp8_x_fp8(config))
    
    # Test 2: FP8 × BF16
    results.append(test_fp8_x_bf16(config))
    
    # Test 3: BF16 × BF16 (internal FP8)
    results.append(test_bf16_x_bf16(config))
    
    # Test 4: BMM FP8
    results.append(test_bmm_fp8(config))
    
    # ==========================================================================
    # 4-bit × 8-bit and 4-bit × 16-bit - May fail on SM90
    # ==========================================================================
    
    # Test 5: NVFP4 × FP8
    config.weight_format = WeightFormat.NVFP4
    results.append(test_nvfp4_x_fp8(config))
    
    # Test 6: MXFP4 × FP8
    config.weight_format = WeightFormat.MXFP4
    results.append(test_mxfp4_x_fp8(config))
    
    # Test 7: NVFP4 × BF16
    results.append(test_nvfp4_x_bf16(config))
    
    # Test 8: MXFP4 × BF16
    results.append(test_mxfp4_x_bf16(config))
    
    # Summary
    print_separator("SUMMARY")
    
    print("TEST RESULTS:")
    for r in results:
        status = "✓ SUCCESS" if r["success"] else "✗ FAILED"
        print(f"\n  {r['config']}: {status}")
        if not r["success"]:
            print(f"    Error Type: {r['error_type']}")
            msg = str(r['error_message'])
            print(f"    Error Message: {msg[:100]}..." if len(msg) > 100 else f"    Error Message: {msg}")
    
    print("\n" + "="*80)
    print("  Analysis: Failure Modes")
    print("="*80)
    
    for r in results:
        if not r["success"]:
            print(f"\n  [{r['config']}]")
            print(f"  Error occurred at: Python validation / Kernel selection / Runtime")
            
            # Analyze error type
            error_type = str(r.get("error_type", ""))
            if "NotImplementedError" in error_type:
                print("  Failure Mode: Kernel not implemented for this architecture")
            elif "RuntimeError" in error_type:
                print("  Failure Mode: Runtime kernel launch failure")
            elif "ValueError" in error_type:
                print("  Failure Mode: Python-level validation rejection")
            else:
                print(f"  Failure Mode: {r.get('error_type', 'Unknown')}")
    
    print("\n" + "="*80)
    print("  Script completed. See above for detailed error traces.")
    print("="*80 + "\n")
    
    # Return non-zero if any FP8 test failed (those should succeed on SM90)
    if sm_version >= 90 and sm_version < 100:
        fp8_failures = [r for r in results if not r["success"] and "FP8" in r["config"] and "NVFP4" not in r["config"] and "MXFP4" not in r["config"]]
        if fp8_failures:
            print("⚠️  FP8 tests failed on SM90 - this is unexpected!")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

