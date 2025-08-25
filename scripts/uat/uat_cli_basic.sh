#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

name="uat_cli_basic"

# 1) medvllm examples
STDOUT1="$WORK_DIR/${name}_examples.out"
code=$(run_cli "$STDOUT1" examples)
if (( code == 0 )); then
  compare_with_golden "$STDOUT1" "$SCRIPT_DIR/expected/cli_examples.golden" || code=1
fi
record_result "examples command" "$code"

# 2) medvllm model list
STDOUT2="$WORK_DIR/${name}_model_list.out"
code=$(run_cli "$STDOUT2" model list)
if (( code == 0 )); then
  compare_with_golden "$STDOUT2" "$SCRIPT_DIR/expected/model_list.golden" || code=1
fi
record_result "model list" "$code"

# Final summary determines exit code
summary
