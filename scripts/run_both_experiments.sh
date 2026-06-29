#!/bin/bash
# Run CH-OPD and FED experiments in parallel on 4 GPUs each.
# CH-OPD uses GPUs 0-3, FED uses GPUs 4-7.
set -e

cd "$(dirname "$0")/.."
mkdir -p outputs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CH_OUTPUT="outputs/ch_opsd_${TIMESTAMP}"
FED_OUTPUT="outputs/fed_opsd_${TIMESTAMP}"

echo "Starting parallel experiments..."
echo "  CH-OPD  → GPUs 0-3 → $CH_OUTPUT"
echo "  FED     → GPUs 4-7 → $FED_OUTPUT"
echo ""

GPUS=0,1,2,3 OUTPUT_DIR="$CH_OUTPUT" bash scripts/run_ch_experiment.sh \
    > "outputs/ch_opsd_${TIMESTAMP}.log" 2>&1 &
PID_CH=$!

GPUS=4,5,6,7 OUTPUT_DIR="$FED_OUTPUT" bash scripts/run_fed_experiment.sh \
    > "outputs/fed_opsd_${TIMESTAMP}.log" 2>&1 &
PID_FED=$!

echo "Launched CH-OPD (PID $PID_CH) and FED (PID $PID_FED)"
echo "Logs: outputs/ch_opsd_${TIMESTAMP}.log  |  outputs/fed_opsd_${TIMESTAMP}.log"
echo ""
echo "Monitor GPU usage: watch -n5 nvidia-smi"
echo "Monitor logs:      tail -f outputs/ch_opsd_${TIMESTAMP}.log"

wait $PID_CH && echo "CH-OPD finished" || echo "CH-OPD FAILED (see log)"
wait $PID_FED && echo "FED finished" || echo "FED FAILED (see log)"

echo ""
echo "Both experiments done."
echo ""
echo "Results:"
[ -f "$CH_OUTPUT/results.json" ] && python -c "
import json; d = json.load(open('$CH_OUTPUT/results.json'))
for r in d: print(f'  CH step {r[\"step\"]}: pass@1={r[\"pass@1\"]:.3f}')
" || echo "  CH-OPD: no results.json"

[ -f "$FED_OUTPUT/results.json" ] && python -c "
import json; d = json.load(open('$FED_OUTPUT/results.json'))
for r in d: print(f'  FED step {r[\"step\"]}: pass@1={r[\"pass@1\"]:.3f}')
" || echo "  FED: no results.json"
