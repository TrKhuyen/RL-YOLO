#!/usr/bin/env bash
set -Eeuo pipefail

KB3_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Current experiment: KB1 inference + four fixed TPE trials + validation report.
# No adaptive policy, tuning retraining, final seed matrix, or automatic test.
exec bash "$KB3_DIR/run_quality_kb3.sh" --stage all "$@"
