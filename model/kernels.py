import torch
import triton
import triton.language as tl

# Try importing hardware-specific libraries (fail gracefully for testing on CPU)
try:
    import deepgemm
    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False

try:
    from flash_mla import flash_mla_with_kvcache
    HAS_FLASH_MLA = True
except ImportError:
    HAS_FLASH_MLA = False

# ---------------------------------------------------------------------------
# DeepGEMM (FP8 Tensor Core MatMul for Blackwell)
# ---------------------------------------------------------------------------
def fp8_matmul(a, b):
    """
    Perform A @ B.T using DeepGEMM FP8 if available.
    Expects `a` and `b` to be scaled float8_e4m3fn.
    """
    if HAS_DEEPGEMM and a.dtype == torch.float8_e4m3fn:
        # deepgemm.gemm natively handles FP8 inputs and scaling
        return deepgemm.gemm(a, b)
    else:
        # Fallback to standard BF16 matmul for testing
        return torch.matmul(a.to(torch.bfloat16), b.t().to(torch.bfloat16))

# ---------------------------------------------------------------------------
# Triton PTX FP8 -> BF16 Conversion
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_to_bf16_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    # Direct hardware conversion via inline PTX (Blackwell SM120/SM100 native)
    out = tl.inline_asm_elementwise(
        "cvt.rn.bf16.fp8e4m3 $0, $1;",
        "=h,h", [x], dtype=tl.bfloat16, is_pure=True, pack=1
    )
    tl.store(out_ptr + offsets, out, mask=mask)

def convert_fp8_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """Uses custom PTX to hardware-convert FP8 to BF16."""
    if not x.is_cuda:
        return x.to(torch.bfloat16)
        
    out = torch.empty_like(x, dtype=torch.bfloat16)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fp8_to_bf16_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out

# ---------------------------------------------------------------------------
# FlashMLA Wrapper
# ---------------------------------------------------------------------------
def decode_flash_mla(q, compressed_kv, page_table, seq_lens):
    """
    Wraps FlashMLA for fast decoding path. 
    `compressed_kv` contains the 160-dim latent representations.
    """
    if HAS_FLASH_MLA:
        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=compressed_kv,
            v_cache=compressed_kv,
            block_table=page_table,
            cache_seqlens=seq_lens,
        )
        return out
    else:
        raise NotImplementedError("FlashMLA required for decoding, but not installed.")
