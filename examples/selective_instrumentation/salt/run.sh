#!/bin/bash
# SALT-FM (C/C++) selective instrumentation checks. Needs `saltfm` on PATH
# (spack load saltfm); exits 77 (SKIP) otherwise.
# Expect lines: "timer: <timer name without the [{file}...] suffix> => yes|no" (yes = instrumented).
# Cases named disabled-*.tau cover behavior SALT-FM does not support yet (see the
# SALT-FM selective-instrumentation parity issue); they are skipped unless SALT_RUN_DISABLED=1.
here=$(cd "$(dirname "$0")" && pwd)
command -v saltfm >/dev/null || { echo "SKIP salt: saltfm not on PATH"; exit 77; }
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
cp "$here/cases/src.cpp" "$work/"
status=0
for sel in "$here"/cases/*.tau; do
  name=$(basename "${sel%.tau}"); exp=${sel%.tau}.expect
  if [[ $name == disabled-* && -z $SALT_RUN_DISABLED ]]; then echo "SKIP salt $name (not supported yet)"; continue; fi
  rm -f "$work/out.cpp"
  (cd "$work" && saltfm --tau_select_file="$sel" --tau_output=out.cpp src.cpp -- >/dev/null 2>"$work/err")
  # SALT writes no output when every file is excluded: nothing is instrumented
  timers=; [ -s "$work/out.cpp" ] && timers=$(grep -o 'TAU_PROFILE_TIMER(tautimer, "[^"]*' "$work/out.cpp" | sed -e 's/.*, "//' -e 's/ *\[{.*//' -e 's/ *$//')
  errs=()
  while IFS= read -r line; do
    [[ $line == timer:\ * ]] || continue
    t=${line#timer: }; t=${t% => *}; want=${line##* => }
    got=no; grep -qxF "$t" <<<"$timers" && got=yes
    [ "$got" = "$want" ] || errs+=("timer '$t': expected $want")
  done < "$exp"
  if [ ${#errs[@]} -eq 0 ]; then echo "PASS salt $name"; else echo "FAIL salt $name"; printf '  %s\n' "${errs[@]}"; status=1; fi
done
exit $status
