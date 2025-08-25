#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

name="uat_cli_inference"

# 1) inference ner --text ... --json-out
STDOUT1="$WORK_DIR/${name}_ner_json.out"
code=$(run_cli "$STDOUT1" inference ner --text "HTN on metformin" --json-out)
if (( code == 0 )); then
  compare_with_golden "$STDOUT1" "$SCRIPT_DIR/expected/inference_ner_json.golden" || code=1
fi
record_result "inference ner --json-out" "$code"

# 2) inference generate from stdin to file
OUTFILE1="$WORK_DIR/${name}_gen.txt"
PROMPT="Explain hypertension in one sentence."
STDOUT2="$WORK_DIR/${name}_generate.out"
# run sending prompt to stdin
set +e
python - <<PY >"$STDOUT2" 2>"$STDOUT2.stderr"
import sys, subprocess
cmd = [sys.executable, "-m", "medvllm.cli", "inference", "generate", "--model", "gpt2", "--output", "${OUTFILE1}"]
proc = subprocess.run(cmd, input=b"${PROMPT}", capture_output=True)
print(proc.stdout.decode("utf-8", "ignore"), end="")
sys.exit(proc.returncode)
PY
code=$?
set -e
# Create a temp golden that checks the output file is non-empty
GOLDEN2="$WORK_DIR/${name}_generate.golden"
echo "FILE_NONEMPTY: ${OUTFILE1}" > "$GOLDEN2"
if (( code == 0 )); then
  compare_with_golden "$STDOUT2" "$GOLDEN2" || code=1
fi
record_result "inference generate -> file" "$code"

# Final summary determines exit code
summary
