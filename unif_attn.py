import triton
import triton.language as tl
import torch

# Pick the FP8 E4M3 dtype for the current platform (AMD uses the "fnuz" variant).
if hasattr(torch, "float8_e4m3fnuz") and torch.version.hip is not None:
    e4m3_dtype = torch.float8_e4m3fnuz
elif hasattr(torch, "float8_e4m3fn"):
    e4m3_dtype = torch.float8_e4m3fn
else:
    e4m3_dtype = torch.float16  # very old torch; FP8 paths unused in the harness

float8_info = torch.finfo(e4m3_dtype)


@triton.jit
def fast_exp(x):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    return tl.math.exp2(x * RCP_LN2)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale: tl.constexpr,  # float32
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    out_scale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ALL_DECODE: tl.constexpr = False,  # bool
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""
    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        # Prescale with RCP_LN2, needed for exp2
        M = (
            tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
            * RCP_LN2
        )

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
    if q_descale_ptr is not None:
        q_descale = tl.load(q_descale_ptr)
        qk_scale = qk_scale * q_descale
    else:
        q_descale = None
    if k_descale_ptr is not None and v_descale_ptr is not None:
        k_descale = tl.load(k_descale_ptr)
        v_descale = tl.load(v_descale_ptr)
        qk_scale = qk_scale * k_descale
    else:
        k_descale = None
        v_descale = None
    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        # to reduce the masking effect when not needed
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
            cache_modifier=KV_cache_modifier,
        )

        K = K_load.to(Q.dtype)

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
            cache_modifier=KV_cache_modifier,
        )

        V = V_load.to(Q.dtype)

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * RCP_LN2

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    if v_descale is not None:
        one_over_L = v_descale / L[:, None]
    else:
        one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L
    if out_scale_ptr is not None:
        acc = acc / tl.load(out_scale_ptr)

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )

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


def _profile():
    """Run under triton-viz profiler (CPU interpreter) and print metrics."""
    from triton_viz.clients import Profiler
    tv = _run_with_triton_viz(Profiler())
    # The profiler prints per-op load/store byte counts and flags issues
    # (non-unrolled loops, inefficient masks, etc.) on shutdown. We also
    # save a trace so it can be opened in the visualizer UI afterwards.
    trace_path = "unif_attn_profile.tvz"
    tv.save(trace_path)
    print(f"[triton-viz] profile trace saved to {trace_path}")
    print(f"    open in UI: triton-visualizer {trace_path}")


def _visualize(port: int = 5001, share: bool = False):
    """Run under triton-viz tracer, save trace, and launch the web UI."""
    from triton_viz.clients import Tracer
    tv = _run_with_triton_viz(Tracer())
    trace_path = "unif_attn_trace.tvz"
    tv.save(trace_path)
    print(f"[triton-viz] trace saved to {trace_path}")
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
