#!/bin/bash
# run_reader.sbatch: example Slurm job for ONE judge-read shard on one GPU with the in-process vLLM reader.
# The reader CLI driver is not part of this repository (it depends on the unpublished judge client); set READER_CMD to your own driver.
# It is called as: $READER_CMD --base BASE --problems ... --texts ... --calls ... --out OUT --workers N --max-num-seqs N [--adapter ...]
# Patterns shown: sha256 pins of the code kit and the input kit, a GPU-class guard (one comparison = one GPU class), max_num_seqs chosen
# by VRAM, a TERM trap that stops the child and reaps only THIS job's GPU processes, resumable output, and the start / END log lines
# that judgelab.stage0_gate parses.
#SBATCH -J judge_read
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@90
#SBATCH --open-mode=append
set -euo pipefail
umask 077
refuse() { echo "JOB_REFUSED $1"; exit 3; }

WORK="${JUDGELAB_ROOT:?}"                 # working area: venv, caches, outputs
CODE="${JUDGELAB_CODE:?}"                 # code kit dir with a SHA256SUMS file
BASE="${BASE_MODEL:?}"                    # base model dir
KIT="${INPUT_KIT:?}"                      # input kit: PROBLEMS.jsonl, TEXTS.jsonl, CALLS_<shard>.jsonl, SHA256SUMS
READER_CMD="${READER_CMD:?}"              # your reader driver
VPY="${VENV_PYTHON:-$WORK/venv/bin/python}"
MODE="${MODE:-HFJ3}"
ARM="${ARM:?}"
SHARD="${SHARD:?}"
[[ "$ARM" =~ ^[A-Za-z0-9_]+$ ]] || refuse arm
[[ "$SHARD" =~ ^[A-Za-z0-9_]+$ ]] || refuse shard

# code and input pins (the SHA256SUMS file itself is pinned, then every listed file is checked)
[[ "$(sha256sum "$CODE/SHA256SUMS" | cut -d' ' -f1)" == "${CODE_SHA256:?}" ]] || refuse code_pin
(cd "$CODE" && sha256sum -c --strict --quiet SHA256SUMS) || refuse code_files
[[ "$(sha256sum "$KIT/SHA256SUMS" | cut -d' ' -f1)" == "${INPUT_SHA256:?}" ]] || refuse input_pin
(cd "$KIT" && sha256sum -c --strict --quiet SHA256SUMS) || refuse input_files
[[ -x "$VPY" ]] || refuse no_venv
CALLS="$KIT/CALLS_$SHARD.jsonl"
[[ -f "$CALLS" ]] || refuse calls_missing

EXTRA=()
if [[ "$ARM" == BASE ]]; then
    [[ -z "${ADAPTER:-}" ]] || refuse base_with_adapter
else
    EXTRA=(--adapter "${ADAPTER:?}" --adapter-weights-sha256 "${ADAPTER_WEIGHTS_SHA256:?}" --adapter-config-sha256 "${ADAPTER_CONFIG_SHA256:?}")
fi

# GPU-class guard: every read of one comparison must come from the same GPU class
GN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ -n "${GPU_NAME:-}" && "$GN" != "$GPU_NAME" ]]; then
    refuse "gpu_class_${GN// /_}_wanted_${GPU_NAME// /_}"
fi
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
SEQS=64; [[ "$MIB" -ge 79000 ]] && SEQS=128      # 64 on 40-48 GB cards, 128 on 80 GB cards

OUT="$WORK/out/${CODE_SHA256:0:12}/${INPUT_SHA256:0:12}/$ARM/$SHARD"
mkdir -p "$OUT/logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 XDG_CACHE_HOME=$WORK/cache VLLM_CACHE_ROOT=$WORK/cache/vllm
export VLLM_USE_FLASHINFER_SAMPLER=0                # required by judgelab.engine.build_engine
echo "== $(date -Is) job=${SLURM_JOB_ID:-none} mode=$MODE gpu=$GN out=$OUT"

reap() {   # kill only GPU processes that belong to THIS job (shared nodes)
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -aqx "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"; then
            kill -9 "$p" 2>/dev/null && echo "reaped own gpu pid $p"
        fi
    done
}
PID=
stop() {
    if [[ -n "$PID" ]]; then
        kill -TERM "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    reap
    echo "TERM: a requeued or resubmitted job resumes from READS.jsonl"
    exit 143
}
trap stop TERM INT

"$VPY" -I -B -X utf8 $READER_CMD --base "$BASE" --problems "$KIT/PROBLEMS.jsonl" --texts "$KIT/TEXTS.jsonl" \
    --calls "$CALLS" --out "$OUT" --workers "$SEQS" --max-num-seqs "$SEQS" --penalty-scope prompt "${EXTRA[@]}" \
    >> "$OUT/logs/reader_${SLURM_JOB_ID:-none}_inc${SLURM_RESTART_COUNT:-0}.log" 2>&1 &
PID=$!
set +e
wait "$PID"
RC=$?
set -e
PID=
reap
echo "== $(date -Is) END rc=$RC out=$OUT"
exit $RC
