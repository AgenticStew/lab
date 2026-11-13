# Is WSL2 / Windows 11 why the 5090 decodes at ~77 tok/s? (13 Nov 2026, ~16:20)

Qwen3.8-27B UD-Q4_K_M, llama.cpp b12189 (WSL2 build), llama-bench -ngl 99 -fa 1 -p 0 -n 256.

- **GPU timestamps (Nsight Systems, prof.nsys-rep):** each 256-token rep spans 3.31-3.35 s on the GPU clock = 76.4-77.3
  tok/s. Inside that span the GPU executes work **95.5%** of the time (~8 CUDA-graph launches per token, ~0.08 ms gap
  between them). Every host-side cost (WSL2's GPU paravirtualisation, the Windows driver model, llama.cpp's own CPU work)
  fits in the other ~4.5%, so a perfect native setup could gain at most ~4-5% (~80 tok/s).
- **The real limit is the GPU work itself:** ~12.4 ms of kernels per token for ~15.4 GB of weights = ~1.24 TB/s effective,
  about 75% of the 1,643 GB/s measured copy bandwidth (1,792 rated). A token is hundreds of small kernels (norms,
  activation quantising, 48 linear-attention state updates, mat-vecs); small kernels don't keep every memory channel busy.
- **Power (smi.csv):** "SW power cap" (575 W) active in 77% of busy samples, SM clock median 2,850 MHz (max 2,895),
  memory clock steady at 13,801 MHz: memory isn't throttled, so the cap costs little here.
- **llama-bench's occasional 95-98 tok/s reps under WSL2 are fake:** 2 of 15 (wsl_r15.json) and 1 of 8 (prof_bench.json)
  reps, but on the GPU's own clock every rep took 3.31-3.35 s. WSL2's system clock jumps (journald logs "Time jumped
  backwards"). Medians of 3 runs plus spread_check.py keep the video's numbers clear of it.
- **Native Windows build not run:** Windows' code-integrity policy (Smart App Control, error 0xc0e90002) blocked the
  downloaded llama.cpp Windows DLLs; not bypassed. The GPU-busy measurement answers the question without it.
  Leftover to delete by hand: C:\spec-bench (16.5 GB model copy + the blocked binaries).
- Earlier finding still stands: the morning's 69 tok/s was Windows apps sharing the machine (~9% slower).
