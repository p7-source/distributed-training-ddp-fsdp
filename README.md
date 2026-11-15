# distributed-training-ddp-fsdp
# Distributed Training on 2x T4: DDP & FSDP

Hands-on experiments with PyTorch DistributedDataParallel (DDP) and Fully Sharded Data Parallel (FSDP) on two NVIDIA T4 GPUs (Kaggle), measuring scaling efficiency and memory savings.

## Setup
- 2x NVIDIA T4 (16 GB each), Kaggle
- GPU link: PHB (over PCIe through the CPU, **no NVLink**), checked with `nvidia-smi topo -m`
- PyTorch DDP and FSDP, NCCL backend
- Timing: 10 warm-up steps, then 50 measured steps

## DDP results: throughput (samples/sec)

| Config | 1 GPU | 2 GPUs | Speedup | Efficiency |
|---|---|---|---|---|
| Batch 256 | 25,319 | 18,910 | 0.75x | 37% |
| Batch 2048 | 35,400 | 60,125 | 1.70x | 85% |
| Batch 2048 + FP16 gradient compression | 35,400 | 64,470 | 1.82x | 91% |

### What happened
- **Small batch got slower on 2 GPUs (0.75x).** Too little compute per step, so the gradient sync over slow PCIe (no NVLink) dominated.
- **Bigger batch fixed it (1.70x).** More compute per step hides the communication cost.
- **FP16 gradient compression pushed it to 91%.** `fp16_compress_hook` halves the bytes sent during all-reduce.

## FSDP results: memory per GPU

| Config | Peak memory per GPU |
|---|---|
| Baseline (no sharding) | 2.70 GB |
| FSDP, whole-model wrap | 2.44 GB |
| FSDP, per-layer wrap (`ModuleWrapPolicy`) | 1.36 GB (**-50%**) |

- Wrapping the whole model as one unit barely helps, because all the parameters are gathered at once.
- Wrapping per layer shards and gathers layer by layer, which halves memory per GPU.

## Key takeaways
- Check the GPU interconnect first: PCIe versus NVLink changes the strategy.
- When the link is slow, scale the batch size and compress gradients.
- The FSDP wrap policy matters as much as using FSDP at all.

## How to run
Open the notebook on Kaggle with **GPU T4 x2** and run all cells.

## What I'd do next
- Test on NVLink GPUs to compare the scaling
- Add mixed-precision training and activation checkpointing to FSDP
