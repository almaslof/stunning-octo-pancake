#!/usr/bin/env bash
# Profile kernel_unified_attention_2d with rocprofv3.
#
# Requires: ROCm >= 6.2 (rocprofv3 shipped then). Tested on MI300X (gfx942).
# Docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
#
# Usage:
#   ./profile_rocprof.sh <mode> [--iters N] [--out DIR]
#
# Modes:
#   trace    kernel-trace → CSV (per-launch timings, grid dims, duration)
#   pftrace  system trace → Perfetto .pftrace (open at https://ui.perfetto.dev)
#   stats    summary stats (sorted by total time, average, etc.)
#   pmc      hardware counters from rocprof_counters.yaml (multi-pass)
#   all      run trace + stats + pftrace + pmc back-to-back
#
# Examples:
#   ./profile_rocprof.sh trace
#   ./profile_rocprof.sh pftrace --iters 20 --out rocprof_out/run1
#   ./profile_rocprof.sh pmc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- parse args ----
MODE="${1:-}"
shift || true

ITERS=10
OUT_DIR="rocprof_out"
APP_ARGS=()  # extra args passed through to bench_unified_attention.py
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iters) ITERS="$2"; shift 2 ;;
        --out)   OUT_DIR="$2"; shift 2 ;;
        --)      shift; APP_ARGS+=("$@"); break ;;
        *)       APP_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "usage: $0 {trace|pftrace|stats|pmc|all} [--iters N] [--out DIR]" >&2
    exit 2
fi

if ! command -v rocprofv3 >/dev/null 2>&1; then
    echo "error: rocprofv3 not found in PATH. Install ROCm >= 6.2 or use the" >&2
    echo "       lmsysorg/sglang rocm image." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

PY=python3
APP=(bench_unified_attention.py profile-run --iters "$ITERS")
if [[ ${#APP_ARGS[@]} -gt 0 ]]; then
    APP+=("${APP_ARGS[@]}")
fi

echo "=== rocprofv3 mode=$MODE out=$OUT_DIR iters=$ITERS ==="

run_trace() {
    echo "--- [trace] kernel-trace CSV ---"
    rocprofv3 \
        --kernel-trace \
        --output-format csv \
        --output-directory "$OUT_DIR/trace" \
        --truncate-kernels \
        -- "$PY" "${APP[@]}"
    echo "  → $OUT_DIR/trace/*.csv"
}

run_pftrace() {
    echo "--- [pftrace] Perfetto system trace ---"
    rocprofv3 \
        --sys-trace \
        --output-format pftrace \
        --output-directory "$OUT_DIR/pftrace" \
        --truncate-kernels \
        -- "$PY" "${APP[@]}"
    echo "  → $OUT_DIR/pftrace/*.pftrace  (open at https://ui.perfetto.dev)"
}

run_stats() {
    echo "--- [stats] summary ---"
    rocprofv3 \
        -S \
        --kernel-trace \
        --output-format csv \
        --output-directory "$OUT_DIR/stats" \
        --truncate-kernels \
        -- "$PY" "${APP[@]}"
    echo "  → $OUT_DIR/stats/*.csv  (domain_stats files have totals per kernel)"
}

run_pmc() {
    echo "--- [pmc] hardware counters (multi-pass via rocprof_counters.yaml) ---"
    # Per-pass `output_directory` inside the YAML wins, but we still set an
    # umbrella dir so any extra artefacts land next to it.
    rocprofv3 \
        -i rocprof_counters.yaml \
        --output-format csv \
        --output-directory "$OUT_DIR/pmc" \
        --truncate-kernels \
        -- "$PY" "${APP[@]}"
    echo "  → $OUT_DIR/pmc/... and rocprof_out/pass*_*/counter_collection.csv"
}

case "$MODE" in
    trace)   run_trace ;;
    pftrace) run_pftrace ;;
    stats)   run_stats ;;
    pmc)     run_pmc ;;
    all)
        run_trace
        run_stats
        run_pftrace
        run_pmc
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac

echo "done."
