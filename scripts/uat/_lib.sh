#!/usr/bin/env bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

UAT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_DIR=${WORK_DIR:-"$UAT_DIR/.tmp"}
mkdir -p "$WORK_DIR"

pass_count=0
fail_count=0

log_info()  { echo -e "${YELLOW}[UAT]${NC} $*"; }
log_pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# Run CLI and capture stdout to a file
run_cli() {
  local outfile="$1"; shift
  # Ensure stable test env
  export CUDA_VISIBLE_DEVICES=""
  export TOKENIZERS_PARALLELISM="false"
  set +e
  python -m medvllm.cli "$@" >"$outfile" 2>"$outfile.stderr"
  local code=$?
  set -e
  echo "$code"
}

# Compare stdout against golden rules
# Golden semantics:
# - Regular line: treated as required substring in stdout
# - Line starting with 'OR: a || b || c' passes if any option is present
# - Line 'FILE_NONEMPTY: <path>': check file exists and non-empty
compare_with_golden() {
  local stdout_file="$1"
  local golden_file="$2"
  local ok=0
  while IFS= read -r line || [ -n "$line" ]; do
    # skip empty/comment
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    if [[ "$line" == OR:* ]]; then
      local opts=${line#OR:}
      local passed=1
      IFS='|' read -ra parts <<< "$opts"
      passed=1
      for raw in "${parts[@]}"; do
        opt=$(echo "$raw" | sed 's/^\s\+//;s/\s\+$//')
        if grep -qF -- "$opt" "$stdout_file"; then
          passed=0
          break
        fi
      done
      if (( passed != 0 )); then
        log_fail "Missing any of alternatives: ${opts}"
        ok=1
      fi
    elif [[ "$line" == FILE_NONEMPTY:* ]]; then
      local fpath=${line#FILE_NONEMPTY:}
      fpath=$(echo "$fpath" | sed 's/^\s\+//;s/\s\+$//')
      if [[ ! -s "$fpath" ]]; then
        log_fail "Expected non-empty file: $fpath"
        ok=1
      fi
    else
      if ! grep -qF -- "$line" "$stdout_file"; then
        log_fail "Missing required substring: $line"
        ok=1
      fi
    fi
  done < "$golden_file"
  return $ok
}

record_result() {
  local name="$1"; local code="$2"
  if (( code == 0 )); then
    pass_count=$((pass_count+1))
    log_pass "$name"
  else
    fail_count=$((fail_count+1))
    log_fail "$name (exit=$code)"
  fi
}

summary() {
  echo
  echo "UAT summary: ${pass_count} passed, ${fail_count} failed"
  if (( fail_count > 0 )); then
    return 1
  fi
  return 0
}
