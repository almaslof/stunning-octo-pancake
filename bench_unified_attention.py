import triton
import torch

# Pick the FP8 E4M3 dtype for the current platform (AMD uses the "fnuz" variant).
if hasattr(torch, "float8_e4m3fnuz") and torch.version.hip is not None:
    e4m3_dtype = torch.float8_e4m3fnuz
elif hasattr(torch, "float8_e4m3fn"):
    e4m3_dtype = torch.float8_e4m3fn
else:
    e4m3_dtype = torch.float16  # very old torch; FP8 paths unused in the harness

float8_info = torch.finfo(e4m3_dtype)
from aiter_unified_attention import kernel_unified_attention_2d

def _make_inputs(
    seq_lens_pairs: list[tuple[int, int]],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
):
    torch.manual_seed(seed)

    query_lens = [p[0] for p in seq_lens_pairs]
    seq_lens = [p[1] for p in seq_lens_pairs]
    num_seqs = len(seq_lens_pairs)
    total_q_tokens = sum(query_lens)
    max_kv_len = max(seq_lens)

    query = torch.randn(total_q_tokens, num_query_heads, head_size, dtype=dtype, device=device)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    value_cache = torch.randn_like(key_cache)
    output = torch.empty_like(query)

    cu_seqlens_q = torch.tensor([0] + query_lens, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, num_blocks, (num_seqs, max_blocks_per_seq), dtype=torch.int32, device=device)

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "output": output,
        "cu_seqlens_q": cu_seqlens_q,
        "seq_lens": seq_lens_t,
        "block_tables": block_tables,
        "query_lens": query_lens,
        "kv_lens": seq_lens,
    }


def _launch_2d(
    inputs: dict,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    scale: float,
    sliding_window: int = 0,
    soft_cap: float = 0.0,
):
    q = inputs["query"]
    k_cache = inputs["key_cache"]
    v_cache = inputs["value_cache"]
    out = inputs["output"]
    cu_seqlens_q = inputs["cu_seqlens_q"]
    seq_lens = inputs["seq_lens"]
    block_tables = inputs["block_tables"]

    num_seqs = seq_lens.shape[0]
    num_queries_per_kv = num_query_heads // num_kv_heads

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    HEAD_SIZE_PADDED = triton.next_power_of_2(head_size)
    TILE_SIZE = block_size  # simple choice; matches BLOCK_SIZE so masking is cheap

    grid = (num_kv_heads, total_num_q_blocks)

    kernel_unified_attention_2d[grid](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=None,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        scale=scale,
        q_descale_ptr=None,
        k_descale_ptr=None,
        v_descale_ptr=None,
        out_scale_ptr=None,
        softcap=soft_cap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_tables.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        USE_ALIBI_SLOPES=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=(soft_cap > 0),
        USE_SINKS=False,
        SLIDING_WINDOW=sliding_window,
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        ALL_DECODE=all(ql == 1 for ql in inputs["query_lens"]),
    )
    return out


# ---------------------------------------------------------------------------
# Benchmarking / profiling / visualization entry points.
#
# Modes (select via `python unif_attn.py <mode>`):
#   bench     - time the kernel on real GPU with mixed prefill+decode batch
#   profile   - run via triton-viz profiler (CPU interpreter), print stats
#   visualize - run via triton-viz tracer (CPU interpreter), launch UI / save
#
# Both `profile` and `visualize` require `pip install triton-viz` and run the
# kernel under the Triton interpreter (CPU), so they use very small inputs.
# ---------------------------------------------------------------------------


# Tiny problem size used by the CPU-interpreter paths (triton-viz).
# Keep everything small — the interpreter is orders of magnitude slower than
# real GPU execution and the trace grows with tile count.
_VIZ_CFG = dict(
    seq_lens_pairs=[(2, 32)],   # 1 sequence, q_len=2, kv_len=32
    num_query_heads=2,
    num_kv_heads=1,             # GQA: 2 queries per KV head
    head_size=16,
    block_size=16,
    num_blocks=4,
    dtype=torch.float32,        # interpreter doesn't support bf16/fp8 well
    sliding_window=0,
    soft_cap=0.0,
)


def _bench():
    """Time the kernel on a real GPU and print TFLOP/s."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "bench mode requires a GPU"

    SEQ_LENS = [(1, 1328), (5, 18), (129, 463)]
    NUM_QUERY_HEADS = 8
    NUM_KV_HEADS = 2
    HEAD_SIZE = 128
    BLOCK_SIZE = 16
    NUM_BLOCKS = 2048
    DTYPE = torch.bfloat16
    SLIDING_WINDOW = 0
    SOFT_CAP = 0.0
    SCALE = HEAD_SIZE ** -0.5

    inputs = _make_inputs(
        SEQ_LENS, NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_SIZE,
        BLOCK_SIZE, NUM_BLOCKS, DTYPE, device,
    )

    def run():
        _launch_2d(
            inputs, NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE,
            scale=SCALE, sliding_window=SLIDING_WINDOW, soft_cap=SOFT_CAP,
        )

    ms = triton.testing.do_bench(run, warmup=25, rep=100)
    total_q = sum(inputs["query_lens"])
    total_kv = sum(inputs["kv_lens"])
    flops = sum(
        4 * NUM_QUERY_HEADS * HEAD_SIZE * q_len * kv_len
        for q_len, kv_len in zip(inputs["query_lens"], inputs["kv_lens"])
    )
    tflops = flops / (ms * 1e-3) / 1e12
    print(
        f"latency: {ms:.3f} ms | "
        f"q_tokens={total_q} kv_tokens={total_kv} | "
        f"~{tflops:.2f} TFLOP/s"
    )


def _run_with_triton_viz(client):
    """Launch `kernel_unified_attention_2d` under the Triton interpreter with
    a triton-viz client attached. Returns the triton_viz module."""
    import triton_viz  # noqa: F401 — imported for side effects / .launch()

    # Wrap the kernel with the desired client. triton-viz intercepts the
    # @triton.jit kernel and re-dispatches via the interpreter.
    traced_kernel = triton_viz.trace(client=client)(kernel_unified_attention_2d)

    cfg = _VIZ_CFG
    num_query_heads = cfg["num_query_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_size = cfg["head_size"]
    block_size = cfg["block_size"]
    num_queries_per_kv = num_query_heads // num_kv_heads
    scale = head_size ** -0.5

    inputs = _make_inputs(
        cfg["seq_lens_pairs"], num_query_heads, num_kv_heads, head_size,
        block_size, cfg["num_blocks"], cfg["dtype"], torch.device("cpu"),
    )

    q = inputs["query"]
    k_cache = inputs["key_cache"]
    v_cache = inputs["value_cache"]
    out = inputs["output"]
    cu_seqlens_q = inputs["cu_seqlens_q"]
    seq_lens = inputs["seq_lens"]
    block_tables = inputs["block_tables"]
    num_seqs = seq_lens.shape[0]

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    HEAD_SIZE_PADDED = triton.next_power_of_2(head_size)
    grid = (num_kv_heads, total_num_q_blocks)

    print(
        f"[triton-viz] interpreter launch grid={grid} "
        f"q_tokens={q.shape[0]} kv_len={inputs['kv_lens']} "
        f"BLOCK_M={BLOCK_M} BLOCK_Q={BLOCK_Q} TILE_SIZE={block_size}"
    )

    traced_kernel[grid](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=None,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        scale=scale,
        q_descale_ptr=None,
        k_descale_ptr=None,
        v_descale_ptr=None,
        out_scale_ptr=None,
        softcap=cfg["soft_cap"],
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_tables.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        USE_ALIBI_SLOPES=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=(cfg["soft_cap"] > 0),
        USE_SINKS=False,
        SLIDING_WINDOW=cfg["sliding_window"],
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        ALL_DECODE=False,
    )
    return triton_viz


def _maybe_save_trace(tv, path: str) -> bool:
    """Try to persist a triton-viz trace; silently skip on older versions."""
    save_fn = getattr(tv, "save", None)
    if save_fn is None:
        print(f"[triton-viz] installed version has no .save(); skipping {path}")
        return False
    save_fn(path)
    print(f"[triton-viz] trace saved to {path}")
    return True


def _profile():
    """Run under triton-viz profiler (CPU interpreter) and print metrics."""
    from triton_viz.clients import Profiler
    tv = _run_with_triton_viz(Profiler())
    # The profiler prints per-op load/store byte counts and flags issues
    # (non-unrolled loops, inefficient masks, etc.) on shutdown.
    _maybe_save_trace(tv, "unif_attn_profile.tvz")


def _visualize(port: int = 5001, share: bool = False):
    """Run under triton-viz tracer, save trace (if supported), and launch UI."""
    from triton_viz.clients import Tracer
    tv = _run_with_triton_viz(Tracer())
    _maybe_save_trace(tv, "unif_attn_trace.tvz")
    print(f"[triton-viz] launching UI on port {port} (share={share}) ...")
    tv.launch(share=share, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="kernel_unified_attention_2d driver")
    parser.add_argument(
        "mode",
        nargs="?",
        default="bench",
        choices=["bench", "profile", "visualize"],
        help="bench: time on GPU; profile/visualize: run under triton-viz on CPU",
    )
    parser.add_argument("--port", type=int, default=5001, help="visualize: UI port")
    parser.add_argument("--share", action="store_true", help="visualize: public share")
    args = parser.parse_args()

    if args.mode == "bench":
        _bench()
    elif args.mode == "profile":
        _profile()
    else:
        _visualize(port=args.port, share=args.share)
