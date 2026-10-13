#!/bin/bash
# Tests the shell-script consumers of the SI format against the common cases:
#   tau_selectfile.sh   file include/exclude lists (file: lines)
#   tau_rewrite         function lists turned into Lua regex lists (func: lines, single-list cases only)
#   tau_compiler.sh     GNU -finstrument-functions-exclude-function-list generation
# The tau_rewrite / tau_compiler.sh code is extracted from the scripts themselves so the real text is tested.
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../../../tools/src
common=$here/../common
status=0
pass() { echo "PASS shell $1"; }
fail() { echo "FAIL shell $1"; shift; for l in "$@"; do echo "  $l"; done; status=1; }

# ---- tau_selectfile.sh --------------------------------------------------------------------------
for sel in "$common"/*.tau; do
  base=${sel%.tau}; name=$(basename "$base")
  grep -q '^file: ' "$base.expect" || continue
  errs=()
  while IFS= read -r line; do
    [[ $line == file:\ * ]] || continue
    f=${line#file: }; f=${f% => *}; want=${line##* => }
    out=$(bash "$src/tau_selectfile.sh" "$sel" "$f"); got=no; [[ $out == OK* ]] && got=yes
    [ "$got" = "$want" ] || errs+=("file '$f': expected $want")
  done < "$base.expect"
  if [ ${#errs[@]} -eq 0 ]; then pass "tau_selectfile.sh $name"; else fail "tau_selectfile.sh $name" "${errs[@]}"; fi
done

# ---- tau_rewrite --------------------------------------------------------------------------------
eval "$(grep '^tau_si_clean()' "$src/tau_rewrite")"
for sel in "$common"/*.tau; do
  base=${sel%.tau}; name=$(basename "$base")
  grep -q '^func: ' "$base.expect" || continue
  inc=false; exc=false
  grep -q '^[[:space:]]*BEGIN_INCLUDE_LIST' "$sel" && inc=true
  grep -q '^[[:space:]]*BEGIN_EXCLUDE_LIST' "$sel" && exc=true
  # tau_rewrite uses either a whitelist or a blacklist, never both
  if $inc && $exc; then echo "SKIP shell tau_rewrite $name (include and exclude lists together)"; continue; fi
  selectfile=$sel; includeroutines=; excluderoutines=
  eval "$(grep -E '^[[:space:]]*(include|exclude)routines=\$\(' "$src/tau_rewrite")"
  mode=none; list=
  $inc && { mode=include; list=$includeroutines; }
  $exc && { mode=exclude; list=$excluderoutines; }
  regexes=()
  if [ -n "$list" ]; then
    # entries look like "re","re"; Lua string escapes (\\) are undone
    while IFS= read -r r; do regexes+=("$r"); done < <(printf '%s' "$list" | awk 'BEGIN{RS="\",\""} {gsub(/^"|"$|\n$/,""); gsub(/\\\\/,"\\"); print}')
  fi
  errs=()
  while IFS= read -r line; do
    [[ $line == func:\ * ]] || continue
    f=${line#func: }; f=${f% => *}; want=${line##* => }
    hit=no
    for r in "${regexes[@]}"; do [[ $f =~ ^($r)$ ]] && { hit=yes; break; }; done
    case $mode in include) got=$hit;; exclude) [ $hit = yes ] && got=no || got=yes;; *) got=yes;; esac
    [ "$got" = "$want" ] || errs+=("func '$f': expected $want")
  done < "$base.expect"
  if [ ${#errs[@]} -eq 0 ]; then pass "tau_rewrite $name"; else fail "tau_rewrite $name" "${errs[@]}"; fi
done

# ---- tau_compiler.sh: GNU exclude-function list ---------------------------------------------------
block=$(awk '/optExcludeFuncsList=\$\(sed -e/{p=1} p{print} p&&/sed .s\/,\$\/\/.\)/{exit}' "$src/tau_compiler.sh")
declare -A gnu=( [01_comments]="foo,bar" [02_exclude_wildcard]="" [05_quoted]="void f(int, int)" \
  [06_metachars]="operator+,a.b,x[0],v|w,cost$" [09_stray_lines]="foo" [10_instrument_section]="foo" [11_empty]="" )
for name in "${!gnu[@]}"; do
  tauSelectFile=$common/$name.tau
  eval "$block"
  want=${gnu[$name]}
  # compiler.sh additionally strips (...) argument lists afterwards
  want=$(echo "$want" | sed -E 's/\([^)]*\)//g'); optExcludeFuncsList=$(echo "$optExcludeFuncsList" | sed -E 's/\([^)]*\)//g')
  if [ "$optExcludeFuncsList" = "$want" ]; then pass "tau_compiler.sh gnu-exclude $name"
  else fail "tau_compiler.sh gnu-exclude $name" "expected '$want' got '$optExcludeFuncsList'"; fi
done
exit $status
