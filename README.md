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
