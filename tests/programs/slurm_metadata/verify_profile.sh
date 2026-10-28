#!/bin/sh
set -eu

program=$1
profile=profile.0.0.0

cleanup() {
  rm -f "$profile"
}
trap cleanup EXIT

has_key() {
  grep -Fq "<name>$1</name>" "$profile"
}

check_present() {
  grep -Fq "<name>$1</name><value>$2</value>" "$profile" ||
    { echo "Missing profile metadata pair: $1=$2" >&2; exit 1; }
}

check_absent() {
  if has_key "$1"; then
    echo "Unexpected profile metadata key: $1" >&2
    exit 1
  fi
}

run_case() {
  rm -f "$profile"
  "$program" "$1"
  test -f "$profile" || { echo "Profile output was not generated for $1" >&2; exit 1; }
}

run_case absent
check_absent SLURM_TOPOLOGY_ADDR
check_absent SLURM_TOPOLOGY_ADDR_PATTERN
check_absent SLURM_LOCALID

run_case complete
check_present SLURM_TOPOLOGY_ADDR root.s2610.frontier10247
check_present SLURM_TOPOLOGY_ADDR_PATTERN switch.switch.node
check_present SLURM_LOCALID 12

run_case partial
check_present SLURM_TOPOLOGY_ADDR root.s2610.frontier10248
check_absent SLURM_TOPOLOGY_ADDR_PATTERN
check_absent SLURM_LOCALID

run_case repeat
check_present SLURM_TOPOLOGY_ADDR root.s2610.frontier10247
check_present SLURM_TOPOLOGY_ADDR_PATTERN switch.switch.node
check_present SLURM_LOCALID 12
for key in SLURM_TOPOLOGY_ADDR SLURM_TOPOLOGY_ADDR_PATTERN SLURM_LOCALID; do
  count=$(grep -o "<name>$key</name>" "$profile" | wc -l)
  test "$count" -eq 1 || { echo "Expected one profile entry for $key, got $count" >&2; exit 1; }
done

run_case equivalent
check_absent SLURM_LOCALID

echo "Slurm metadata collection/profile tests passed."
