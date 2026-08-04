# RTBQSim Project Overview

This project uses **RT-Core** for gate fusion and **ELL** for batched state-vector updates.
This document summarizes the project structure, execution workflow, and compares results with NVIDIA cuQuantum.

---
## Origins and Acknowledgements

This project is originally forked from and inspired by the following repositories:
- **BQSim**: https://github.com/IDEA-CUHK/BQSim.git
- **RTSpMSpM**: https://github.com/escalab/RTSpMSpM.git

---

## Our work and requirement
- Stage 1 = doing gate fusion and produce the fused matrix
- Stage 2 = compute BQCS with ell format fused matrix

### Build Requirements and Environment
Current default dependencies in `rt_compile.sh` (overridable via environment variables):
- **CUDA Toolkit** (including `nvcc` and matching driver)
- **OptiX SDK 9.0.0** (`OptiX_INSTALL_DIR` should point to the OptiX 9.0.0 SDK path)
- **GCC/G++** (default `/usr/bin/gcc-9`, `/usr/bin/g++-9`)
- **OpenMP** (default `libgomp`)
- **cuQuantum** (`CUQUANTUM_ROOT`; adjust if not required in your setup)

Current experimental GPU environment:
- **GPU**: NVIDIA RTX PRO 4000 Blackwell
- **VRAM**: 24467 MiB, approximately 24 GB
- The full benchmark suite is evaluated under this 24 GB VRAM setup. Smaller circuits may run with less VRAM, but the current reported results assume this GPU memory capacity.

---

## RTBQSim Execution Entry
The main executable starts from `RTBQSim/apps/RTBQSim.cu`.

At runtime, `RTBQSim.cu`:
- parses command-line arguments such as `--file`, `--batch_size`, `--num_batch`, `--ps`, and `--pv`;
- loads the input QASM circuit into `QuantumComputation`;
- creates `QBatchSimulator` with the selected batch size and number of batches;
- calls `QBatchSimulator::simulate()` to run Stage 1 gate fusion and Stage 2 ELL-based BQCS;
- checks whether replicated batch states remain consistent after simulation;
- optionally writes the output state vector and prints JSON-format simulation statistics.

### Numeric Precision Policy
- Stage-1/Stage-2 simulation numeric type is fixed to `fp64`.
- Ray-hit geometry path keeps OptiX-required float-based geometry representation (`fp32`) where required by API/data layout.

---


## Execution Flow (Current Script Behavior)
`rt_bqsim.sh` runs multiple QASM circuit sets in sequence (for example: tsp/vqe/qnn/graph_state/portfolio), with fixed settings per case:

- `--ps --pv`
- `--batch_size 32`
- `--num_batch 50`

This is used for performance evaluation and state output generation on the current RT gate-fusion path.

---

## Run Without Docker
(Simulation precision is fixed to fp64.)
```bash
cd RTBQSim
bash rt_compile.sh
bash rt_bqsim.sh
```

## Run With Docker
Method 1. Build image and enter container (interactive mode)
```bash
./run_docker.sh --build
```
Then run inside container:
```bash
bash RTBQSim/rt_compile.sh
bash RTBQSim/rt_bqsim.sh
```

Method 2. Auto-run compile + execute inside container (`rt_compile.sh` + `rt_bqsim.sh`)
```bash
./run_docker.sh --auto-run
```

---

## Baseline (cuQuantum) Workflow
The project keeps a cuQuantum baseline path for comparison against `RTBQSim`.

- Compile baseline target:
```bash
bash RTBQSim/cuquantum_compile.sh
```
- Run baseline batch suite:
```bash
bash RTBQSim/cuquantum.sh
```
### Docker usage for baseline
Run inside container:
```bash
bash RTBQSim/cuquantum_compile.sh
bash RTBQSim/cuquantum.sh
```
Or directly from host through docker runner:
```bash
./run_docker.sh -- bash -lc "bash RTBQSim/cuquantum_compile.sh && bash RTBQSim/cuquantum.sh"
```

---

## Optional Qiskit Workflow
Qiskit is kept as an optional path for extra verification and fused-gate export. 

Run:
```bash
bash RTBQSim/qiskit.sh
```

`qiskit.sh` currently:
- runs Qiskit Aer no-fusion checks on GPU and multi-core CPU;
- exports Qiskit fused-gate files to `RTBQSim/log/fused_gates/qiskit_<circuit>_n<qubits>.txt`.

Python requirements:
- `qiskit`
- `qiskit_aer`
- `numpy`

The script first tries `RTBQSim/.venv/bin/python`; otherwise it falls back to `python3`.

---

## Optional Qiskit Fusion + cuQuantum Workflow
This optional path uses Qiskit to export fused gates and cuQuantum to simulate those fused gates.

Run:
```bash
bash RTBQSim/qiskit_cuquantum.sh
```

Main outputs:
- `RTBQSim/log/qiskit_cuquantum_reuse.txt`
- `RTBQSim/log/fused_gates/qiskit_<circuit>_n<qubits>.txt`

---

## Outputs and Logs
Common output paths (may vary slightly by runtime options):
- `RTBQSim/log/`: run outputs and state dumps (for example, `log/results/state/*.txt`)
- `RTBQSim/log/fused_gates/`: optional Qiskit fused gate export files
- `RTBQSim/build-rt/`: build artifacts
