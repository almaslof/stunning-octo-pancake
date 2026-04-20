Downloaded from: https://triton-lang.org/main/getting-started/tutorials/index.html

## Set-Up
```
git clone git@github.com:almaslof/stunning-octo-pancake.git learn_triton

alias drun='docker run -it --rm --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $(pwd):/persistent --shm-size 256G -e HIP_VISIBLE_DEVICES=0,1,2,3'
drun --name tttal lmsysorg/sglang:v0.5.7-rocm700-mi30x bash

```

## To debug
```
TRITON_INTERPRET=1 python3 ttt.py
import pdb; pdb.set_trace()
```

## Debug, profile, vizualize
https://github.com/Deep-Learning-Profiling-Tools/triton-viz


```
python3 bench_unified_attention.py visualize --port 5001
python3 bench_unified_attention.py profile
python3 bench_unified_attention.py bench
```

## Profile on AMD GPU with rocprofv3
Docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html

```
# per-launch kernel trace (CSV)
./profile_rocprof.sh trace

# perfetto timeline (open at https://ui.perfetto.dev)
./profile_rocprof.sh pftrace

# summary stats
./profile_rocprof.sh stats

# multi-pass hardware counters (occupancy / VALU / MFMA / memory / cache)
./profile_rocprof.sh pmc

# or everything at once
./profile_rocprof.sh all --iters 20 --out rocprof_out/run1
```

Outputs go to `rocprof_out/<mode>/` (per-pass counters land in
`rocprof_out/passN_*` as declared in `rocprof_counters.yaml`).
