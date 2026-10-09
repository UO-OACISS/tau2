#!/bin/bash
# Runs every <consumer>/run.sh. Exit code 77 from a harness means SKIP.
here=$(cd "$(dirname "$0")" && pwd)
pass=0; fail=0; skip=0
for s in "$here"/*/run.sh; do
  d=$(basename "$(dirname "$s")")
  out=$(bash "$s" 2>&1); rc=$?
  echo "$out" | grep -v '^PASS'
  case $rc in 0) pass=$((pass+1)); echo "== $d: OK";; 77) skip=$((skip+1)); echo "== $d: SKIP";; *) fail=$((fail+1)); echo "== $d: FAILED";; esac
done
echo "consumers ok=$pass failed=$fail skipped=$skip"
[ $fail -eq 0 ]
