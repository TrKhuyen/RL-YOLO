#!/usr/bin/env bash
set -Eeuo pipefail

KB3_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Current experiment: KB1 inference + TPE from complete pretrained KB1 recipe + report.
# No adaptive policy, tuning retraining, final seed matrix, or automatic test.
exec bash "$KB3_DIR/run_quality_kb3.sh" --stage all "$@"
