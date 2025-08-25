#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

RC=0

log_info "Running UAT scenarios..."

if bash "$SCRIPT_DIR/uat_cli_basic.sh"; then
  :
else
  RC=1
fi

if bash "$SCRIPT_DIR/uat_cli_inference.sh"; then
  :
else
  RC=1
fi

exit $RC
