#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QISKIT_DIR="${ROOT_DIR}/qiskit_test"
CUQ_DIR="${ROOT_DIR}/build/cuquantum_test"
VENV_DIR="${ROOT_DIR}/.venv"
CUQUANTUM_VENV_LIB="${VENV_DIR}/lib/python3.12/site-packages/cuquantum/lib"
CUTENSOR_VENV_LIB="${VENV_DIR}/lib/python3.12/site-packages/cutensor/lib"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

append_ld_library_path() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
      export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH}"
    else
      export LD_LIBRARY_PATH="${dir}"
    fi
  fi
}

: "${CUQ_BATCH_SIZE:=32}"
: "${CUQ_NUM_BATCH:=50}"
: "${CUQ_OUTPUT_STATE:=0}"
: "${CUQ_BINARY:=${CUQ_DIR}/cuquantum}"
: "${CUQ_FIDELITY_TEST:=1}"
: "${QISKIT_PYTHON_BIN:=}"
: "${QISKIT_REFERENCE_DEVICE:=cpu}"
: "${QISKIT_REFERENCE_FUSION:=0}"
: "${QISKIT_REFERENCE_ROUNDS:=1}"
: "${QISKIT_CPU_THREADS:=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)}"
: "${FIDELITY_THRESHOLD:=0.9999}"
: "${RMSE_THRESHOLD:=1e-5}"

append_ld_library_path "${CUQUANTUM_VENV_LIB}"
append_ld_library_path "${CUTENSOR_VENV_LIB}"

mkdir -p "${ROOT_DIR}/log/results/state"

if [[ ! -x "${CUQ_BINARY}" ]]; then
  echo "[cuquantum.sh] No usable cuQuantum binary found: ${CUQ_BINARY}" >&2
  echo "[cuquantum.sh] Build it with bash cuquantum_compile.sh or set CUQ_BINARY." >&2
  exit 1
fi

check_python_modules() {
  local python_bin="$1"
  shift
  "${python_bin}" - "$@" <<'PY'
import importlib
import sys

missing = []
for module in sys.argv[1:]:
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(module)
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))
PY
}

choose_python_bin() {
  local -a candidates=()
  if [[ -n "${QISKIT_PYTHON_BIN}" ]]; then
    candidates+=("${QISKIT_PYTHON_BIN}")
  fi
  candidates+=("${VENV_DIR}/bin/python" "python3" "python")

  local candidate
  local seen="|"
  for candidate in "${candidates[@]}"; do
    [[ -z "${candidate}" ]] && continue
    if [[ "${seen}" == *"|${candidate}|"* ]]; then
      continue
    fi
    seen="${seen}${candidate}|"
    if command -v "${candidate}" >/dev/null 2>&1 \
      && check_python_modules "${candidate}" qiskit qiskit_aer numpy >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN=""
if [[ "${CUQ_FIDELITY_TEST}" == "1" ]]; then
  PYTHON_BIN="$(choose_python_bin || true)"
  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "[cuquantum.sh] Fidelity testing requires qiskit, qiskit_aer, and numpy." >&2
    echo "[cuquantum.sh] Set QISKIT_PYTHON_BIN=/path/to/python or install them in ${VENV_DIR}." >&2
    exit 1
  fi
fi

extract_ms() {
  local label="$1"
  local output="$2"
  printf '%s\n' "${output}" | awk -v prefix="${label}: " '
    index($0, prefix) == 1 { print $(NF-1) }
  ' | tail -n1
}

extract_value() {
  local label="$1"
  local output="$2"
  printf '%s\n' "${output}" | awk -v prefix="${label}: " '
    index($0, prefix) == 1 {
      sub(prefix, "", $0)
      print $0
    }
  ' | tail -n1
}

run_qiskit_reference() {
  local circuit="$1"
  local qubits="$2"
  "${PYTHON_BIN}" "${QISKIT_DIR}/qiskit_test.py" \
    --circuit_name "${circuit}" \
    --num_qubits "${qubits}" \
    --device "${QISKIT_REFERENCE_DEVICE}" \
    --fusion "${QISKIT_REFERENCE_FUSION}" \
    --rounds "${QISKIT_REFERENCE_ROUNDS}" \
    --max-parallel-threads "${QISKIT_CPU_THREADS}" \
    --output-suffix "cuquantum_raw_reference"
}

run_case() {
  local circuit="$1"
  local qubits="$2"
  local output_state="${CUQ_OUTPUT_STATE}"
  local cuquantum_output
  local qiskit_output=""
  local qiskit_runtime=""
  local qiskit_state=""
  local comparison_output=""
  local comparison_status=0
  local cuquantum_state="${ROOT_DIR}/log/results/state/cuquantum${circuit}_n${qubits}.txt"

  if [[ "${CUQ_FIDELITY_TEST}" == "1" ]]; then
    output_state=1
  fi

  cuquantum_output="$(
    cd "${CUQ_DIR}"
    "${CUQ_BINARY}" "${circuit}" "${qubits}" "${CUQ_BATCH_SIZE}" "${CUQ_NUM_BATCH}" 0 "${output_state}"
  )"
  printf '%s\n' "${cuquantum_output}"

  if [[ "${CUQ_FIDELITY_TEST}" != "1" ]]; then
    echo
    return 0
  fi

  qiskit_output="$(run_qiskit_reference "${circuit}" "${qubits}")"
  qiskit_runtime="$(extract_ms "Qiskit runtime" "${qiskit_output}")"
  qiskit_state="$(extract_value "Output saved" "${qiskit_output}")"
  if [[ -z "${qiskit_runtime}" || -z "${qiskit_state}" ]]; then
    echo "[cuquantum.sh] Failed to run Qiskit reference for ${circuit}_n${qubits}." >&2
    return 1
  fi

  if comparison_output="$(
    "${PYTHON_BIN}" "${QISKIT_DIR}/compare_statevectors.py" \
      --qiskit "${qiskit_state}" \
      --cuquantum "${cuquantum_state}" \
      --fidelity-threshold "${FIDELITY_THRESHOLD}" \
      --rmse-threshold "${RMSE_THRESHOLD}"
  )"; then
    comparison_status=0
  else
    comparison_status=$?
  fi

  echo "Qiskit reference runtime: ${qiskit_runtime} [ms]"
  echo "${comparison_output}"
  echo
  if (( comparison_status != 0 )); then
    return "${comparison_status}"
  fi
}

run_suite() {
  if (( $# > 0 )); then
    if (( $# % 2 != 0 )); then
      echo "[cuquantum.sh] Usage: bash cuquantum.sh [circuit qubits]..." >&2
      return 1
    fi
    while (( $# > 0 )); do
      run_case "$1" "$2"
      shift 2
    done
    return 0
  fi

  run_case tsp 16
  run_case vqe 12
  run_case vqe 14
  run_case vqe 16
  run_case qv 12
  run_case qv 14
  run_case qaoa 13
  run_case qaoa 15
  run_case qft 14
  run_case qft 16
  run_case qft 18
  run_case portfolio_vqe 16
  run_case portfolio_vqe 17
  run_case portfolio_vqe 18
  run_case graph_state 16
  run_case graph_state 18
  run_case graph_state 20
  run_case qnn 17
  run_case qnn 19
  run_case qnn 21
}

run_suite "$@"
