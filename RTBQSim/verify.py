import argparse
import subprocess
from pathlib import Path

import numpy as np


def load_statevector(path):
    data = np.loadtxt(path, dtype=np.float64)

    if data.ndim == 2:
        if data.shape[1] != 2:
            raise ValueError(f"Unsupported 2D shape {data.shape} in {path}")
        vec = data[:, 0] + 1j * data[:, 1]
        return vec.astype(np.complex128)

    if data.ndim == 1:
        if data.size % 2 == 0:
            return (data[0::2] + 1j * data[1::2]).astype(np.complex128)
        return data.astype(np.complex128)

    raise ValueError(f"Unsupported data shape {data.shape} in {path}")


def normalize(vec, name):
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0.0):
        raise ValueError(f"{name} statevector has zero norm")
    return vec / norm


def build_cuquantum_reference(
    circuit_name,
    num_qubits,
    batch_size,
    num_batch,
    cuquantum_binary,
    reference_path,
    reuse_reference,
):
    if reuse_reference and reference_path.exists():
        return 0

    if not cuquantum_binary.exists():
        print(f"Missing cuQuantum binary: {cuquantum_binary}")
        print("Build it first with: bash cuquantum_compile.sh")
        return 1

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cuquantum_binary),
        circuit_name,
        str(num_qubits),
        str(batch_size),
        str(num_batch),
        "0",  # raw circuit path, no Qiskit fused gate file
        "1",  # output state file for fidelity verification
    ]
    result = subprocess.run(
        cmd,
        cwd=cuquantum_binary.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("cuQuantum reference generation failed.")
        if result.stdout.strip():
            print("cuQuantum stdout:")
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print("cuQuantum stderr:")
            print(result.stderr.rstrip())
        return result.returncode

    if not reference_path.exists():
        print(f"cuQuantum did not create expected reference file: {reference_path}")
        return 1

    return 0


def verify(
    circuit_name,
    num_qubits,
    qbsim_path,
    reference_path,
    fidelity_threshold,
    rmse_threshold,
):
    for p in (qbsim_path, reference_path):
        if not p.exists():
            print(f"Missing file: {p}")
            return 1

    print("=======================================")
    print(f"benchmark:    {circuit_name}_n{num_qubits}")
    print("Reference:    cuQuantum")

    v_sim = load_statevector(qbsim_path)
    v_ref = load_statevector(reference_path)

    if len(v_sim) != len(v_ref):
        print(f"Length mismatch: RTBQSim={len(v_sim)} vs cuQuantum={len(v_ref)}")
        return 1

    v_sim = normalize(v_sim, "RTBQSim")
    v_ref = normalize(v_ref, "cuQuantum")

    overlap = np.vdot(v_sim, v_ref)
    if np.abs(overlap) > 0:
        v_sim_aligned = v_sim * np.exp(1j * np.angle(overlap))
    else:
        v_sim_aligned = v_sim

    fidelity = np.clip(np.abs(np.vdot(v_ref, v_sim)) ** 2, 0.0, 1.0)
    rmse = np.sqrt(np.mean(np.abs(v_ref - v_sim_aligned) ** 2))
    max_abs_err = np.max(np.abs(v_ref - v_sim_aligned))

    print(f"Fidelity:     {fidelity:.12f}")
    print(f"RMSE:         {rmse:.6e}")
    print(f"Max abs diff: {max_abs_err:.6e}")

    passed = fidelity >= fidelity_threshold and rmse <= rmse_threshold
    if passed:
        print("PASS")
        return 0

    print(
        f"FAIL (thresholds: fidelity>={fidelity_threshold}, rmse<={rmse_threshold})"
    )
    return 2


def main():
    parser = argparse.ArgumentParser(description="Verify RTBQSim statevector with cuQuantum.")
    parser.add_argument("-c", "--circuit", type=str, required=True)
    parser.add_argument("-n", "--qubits", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batch", type=int, default=50)
    parser.add_argument("--fidelity-threshold", type=float, default=0.9999)
    parser.add_argument("--rmse-threshold", type=float, default=1e-5)
    parser.add_argument("--qbsim", type=str, default=None)
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--cuquantum-binary", type=str, default=None)
    parser.add_argument(
        "--reuse-reference",
        action="store_true",
        help="Use an existing cuQuantum reference state file instead of regenerating it.",
    )
    # Kept for backward compatibility with older verify.py calls; cuQuantum does not use it.
    parser.add_argument("--device", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    qbsim_path = (
        Path(args.qbsim)
        if args.qbsim
        else base / "log" / "results" / "state" / f"qbsim_{args.circuit}_n{args.qubits}.txt"
    )
    reference_path = (
        Path(args.reference)
        if args.reference
        else base / "log" / "results" / "state" / f"cuquantum{args.circuit}_n{args.qubits}.txt"
    )
    cuquantum_binary = (
        Path(args.cuquantum_binary)
        if args.cuquantum_binary
        else base / "build" / "cuquantum_test" / "cuquantum"
    )

    ref_status = build_cuquantum_reference(
        circuit_name=args.circuit,
        num_qubits=args.qubits,
        batch_size=args.batch_size,
        num_batch=args.num_batch,
        cuquantum_binary=cuquantum_binary,
        reference_path=reference_path,
        reuse_reference=args.reuse_reference,
    )
    if ref_status != 0:
        return ref_status

    return verify(
        circuit_name=args.circuit,
        num_qubits=args.qubits,
        qbsim_path=qbsim_path,
        reference_path=reference_path,
        fidelity_threshold=args.fidelity_threshold,
        rmse_threshold=args.rmse_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
