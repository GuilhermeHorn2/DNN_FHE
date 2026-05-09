#!/usr/bin/env bash
# run_all_variants_h30.sh — sweep DiNN FHE inference across
#   {MNIST_signal, MNIST_heaviside, MNIST_relu}
#   x {hidden_size 30}
#
# Each run is sequential (each instance needs ~10 GB RAM for bootstrap keys).
# Per-run output is captured to its own log file under run_logs/<timestamp>/.
#
# Usage:
#   ./run_all_variants_h30.sh                       # uses ./test_data
#   ./run_all_variants_h30.sh /path/to/mnist        # custom data dir
#
# Tip: 3 runs at ~5.6h each is ~17h total. Run under nohup/tmux:
#   nohup ./run_all_variants_h30.sh &> sweep.out &
#   tmux new -s fhe ./run_all_variants_h30.sh

set -uo pipefail

# ---------- config ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/data/mnist}"
# DATA_DIR="${1:-${SCRIPT_DIR}/test_data}"
LOG_DIR="${SCRIPT_DIR}/run_logs/$(date +%Y%m%d_%H%M%S)"

VARIANTS=(
    MNIST_signal
    MNIST_heaviside
    MNIST_relu
)
HIDDEN_SIZES=(30)

# ---------- sanity ----------
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: data directory not found: $DATA_DIR" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

echo "Script dir : $SCRIPT_DIR"
echo "Data dir   : $DATA_DIR"
echo "Log dir    : $LOG_DIR"
echo "Variants   : ${VARIANTS[*]}"
echo "Sizes      : ${HIDDEN_SIZES[*]}"
echo

# ---------- helpers ----------
fmt_hms() {
    local s=$1
    printf "%dh %02dm %02ds" $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

# ---------- run sweep ----------
declare -a RESULTS
TOTAL_START=$SECONDS

for variant in "${VARIANTS[@]}"; do
    VARIANT_DIR="${SCRIPT_DIR}/${variant}"
    BUILD_DIR="${VARIANT_DIR}/build"
    BIN="${BUILD_DIR}/main"

    if [[ ! -d "$VARIANT_DIR" ]]; then
        echo "[SKIP] $variant — directory not found"
        RESULTS+=("$variant (any): SKIPPED — no directory")
        continue
    fi

    # Binary must already exist — build is the user's responsibility.
    if [[ ! -x "$BIN" ]]; then
        echo "[SKIP] $variant — binary not found at $BIN"
        echo "       build it with: cmake -B $BUILD_DIR -S $VARIANT_DIR && cmake --build $BUILD_DIR -j4"
        RESULTS+=("$variant (any): SKIPPED — not built")
        continue
    fi

    for size in "${HIDDEN_SIZES[@]}"; do
        LOG_FILE="${LOG_DIR}/${variant}_h${size}.log"
        TAG="${variant} h=${size}"

        echo
        echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Running $TAG ==="
        echo "  log: $LOG_FILE"
        echo "  monitor: tail -f $LOG_FILE"

        START=$SECONDS
        if (cd "$BUILD_DIR" && ./main "$DATA_DIR" "$size") &> "$LOG_FILE"; then
            ELAPSED=$((SECONDS - START))
            echo "  done in $(fmt_hms $ELAPSED)"
            RESULTS+=("$TAG: OK   ($(fmt_hms $ELAPSED))")
        else
            RC=$?
            ELAPSED=$((SECONDS - START))
            echo "  FAILED (rc=$RC) after $(fmt_hms $ELAPSED) — see $LOG_FILE"
            RESULTS+=("$TAG: FAIL ($(fmt_hms $ELAPSED), rc=$RC)")
        fi
    done
done

# ---------- summary ----------
TOTAL=$((SECONDS - TOTAL_START))
echo
echo "================ SUMMARY ================"
echo "Total wall time: $(fmt_hms $TOTAL)"
echo
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo
echo "Logs in: $LOG_DIR"