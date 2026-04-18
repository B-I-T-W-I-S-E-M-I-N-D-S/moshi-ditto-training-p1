#!/usr/bin/env bash
# =============================================================================
# prepare_and_train.sh — Complete Bridge-Ditto Training Pipeline
# =============================================================================
# End-to-end script for RunPod:
#   1. Prepare HDTF data (video → motion/eye/emo features) using Ditto's pipeline
#   2. Extract bridge audio features (replaces HuBERT extraction)
#   3. Gather training data list
#   4. Launch distributed training
#
# Prerequisites:
#   - Run scripts/setup_environment.sh first
#   - HDTF dataset available at $HDTF_ROOT
#   - Ditto checkpoints at ditto-train/checkpoints/ditto_pytorch/
#
# Usage:
#   bash scripts/prepare_and_train.sh /workspace/HDTF
#
# Or with custom settings:
#   HDTF_ROOT=/data/hdtf NUM_GPUS=4 EPOCHS=500 bash scripts/prepare_and_train.sh
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
HDTF_ROOT="${1:-${HDTF_ROOT:-/workspace/HDTF}}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-1000}"
LR="${LR:-1e-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bridge_ditto_hdtf_v1}"
AUDIO_FEAT_DIM="${AUDIO_FEAT_DIM:-1103}"
MOTION_FEAT_DIM="${MOTION_FEAT_DIM:-265}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DITTO_TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DITTO_TRAIN_DIR")"

DITTO_PYTORCH_PATH="${DITTO_TRAIN_DIR}/checkpoints/ditto_pytorch"
BRIDGE_CKPT="${PROJECT_ROOT}/checkpoints/bridge_best.pt"
BRIDGE_CONFIG="${PROJECT_ROOT}/bridge_module/config.yaml"

# Output paths
DATA_INFO_JSON="${HDTF_ROOT}/data_info.json"
DATA_LIST_JSON="${HDTF_ROOT}/bridge_data_list_train.json"
DATA_PRELOAD_PKL="${HDTF_ROOT}/bridge_preload.pkl"

BOLD="\033[1m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
RED="\033[1;31m"
RESET="\033[0m"

log() { echo -e "${CYAN}[pipeline]${RESET} $*"; }
ok()  { echo -e "${GREEN}[pipeline]${RESET} ✅  $*"; }
err() { echo -e "${RED}[pipeline]${RESET} ❌  $*"; exit 1; }

SECONDS=0

echo -e "${BOLD}"
echo "═══════════════════════════════════════════════════════════════"
echo "   Bridge-Ditto Training Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "   HDTF Root    : $HDTF_ROOT"
echo "   GPUs         : $NUM_GPUS"
echo "   Batch/GPU    : $BATCH_SIZE"
echo "   Epochs       : $EPOCHS"
echo "   Experiment   : $EXPERIMENT_NAME"
echo "   Audio dim    : $AUDIO_FEAT_DIM"
echo "   Motion dim   : $MOTION_FEAT_DIM"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${RESET}"

# ── Validate paths ────────────────────────────────────────────────────────────
[ -d "$HDTF_ROOT" ] || err "HDTF dataset not found: $HDTF_ROOT"
[ -f "$BRIDGE_CKPT" ] || err "Bridge checkpoint not found: $BRIDGE_CKPT"
[ -f "$BRIDGE_CONFIG" ] || err "Bridge config not found: $BRIDGE_CONFIG"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Video Processing (reuse Ditto's prepare_data.sh)
# Extracts: motion features, eye features, emotion features
# ══════════════════════════════════════════════════════════════════════════════
log "PHASE 1: Video Processing (Ditto data preparation)..."

if [ -f "$DATA_INFO_JSON" ]; then
    ok "data_info.json already exists, skipping video processing."
    ok "  (Delete $DATA_INFO_JSON to re-run video processing)"
else
    log "Running Ditto's video processing pipeline..."
    log "(This extracts motion, eye, and emotion features from videos)"

    HUBERT_ONNX="${DITTO_PYTORCH_PATH}/aux_models/hubert_streaming_fix_kv.onnx"
    MP_FACE_LMK_TASK="${DITTO_PYTORCH_PATH}/aux_models/face_landmarker.task"

    cd "${DITTO_TRAIN_DIR}/prepare_data"

    # Check Ditto checkpoints
    python scripts/check_ckpt_path.py --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"

    # Crop video
    python scripts/crop_video_by_LP.py -i "${DATA_INFO_JSON}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"

    # Extract audio from video
    python scripts/extract_audio_from_video.py -i "${DATA_INFO_JSON}"

    # Motion features (LivePortrait)
    python scripts/extract_motion_feat_by_LP.py -i "${DATA_INFO_JSON}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"
    python scripts/extract_motion_feat_by_LP.py -i "${DATA_INFO_JSON}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" --flip_flag

    # Eye features (MediaPipe)
    python scripts/extract_eye_ratio_from_video.py -i "${DATA_INFO_JSON}" \
        --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}"
    python scripts/extract_eye_ratio_from_video.py -i "${DATA_INFO_JSON}" \
        --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" --flip_lmk_flag

    # Emotion features
    python scripts/extract_emo_feat_from_video.py -i "${DATA_INFO_JSON}"

    cd "${DITTO_TRAIN_DIR}"

    ok "Video processing complete."
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Bridge Audio Feature Extraction (replaces HuBERT)
# Audio → Mimi → Bridge → .npy features (T, 1024) @ 25 Hz
# ══════════════════════════════════════════════════════════════════════════════
log "PHASE 2: Bridge Audio Feature Extraction..."
log "  (Replaces HuBERT ONNX extraction with Mimi → Bridge pipeline)"

cd "${DITTO_TRAIN_DIR}"

if [ "$NUM_GPUS" -gt 1 ]; then
    log "Multi-GPU extraction on $NUM_GPUS GPUs..."

    # Launch one process per GPU in parallel
    pids=()
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$gpu_id python preprocess_bridge_features.py \
            -i "${DATA_INFO_JSON}" \
            --bridge_ckpt "${BRIDGE_CKPT}" \
            --bridge_config "${BRIDGE_CONFIG}" \
            --device "cuda" \
            --num_gpus "${NUM_GPUS}" \
            --gpu_id "${gpu_id}" \
            --output_key "bridge_aud_npy_list" &
        pids+=($!)
        log "  GPU $gpu_id: PID ${pids[-1]}"
    done

    # Wait for all processes
    for pid in "${pids[@]}"; do
        wait "$pid" || err "Bridge extraction failed on one GPU (PID: $pid)"
    done
else
    python preprocess_bridge_features.py \
        -i "${DATA_INFO_JSON}" \
        --bridge_ckpt "${BRIDGE_CKPT}" \
        --bridge_config "${BRIDGE_CONFIG}" \
        --device "cuda" \
        --output_key "bridge_aud_npy_list"
fi

ok "Bridge feature extraction complete."

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Gather Training Data List
# Creates the data_list_json that Stage2Dataset reads
# ══════════════════════════════════════════════════════════════════════════════
log "PHASE 3: Gathering training data list..."

cd "${DITTO_TRAIN_DIR}/prepare_data"

python scripts/gather_data_list_json_for_train.py \
    -i "${DATA_INFO_JSON}" \
    -o "${DATA_LIST_JSON}" \
    --aud_feat_name "bridge_aud_npy_list" \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --with_flip

ok "Training data list created: ${DATA_LIST_JSON}"

# ── Optional: preload data into pickle for faster training ────────────────────
log "Creating preloaded data pickle (optional, speeds up training)..."

python scripts/preload_train_data_to_pkl.py \
    --data_list_json "${DATA_LIST_JSON}" \
    --data_preload_pkl "${DATA_PRELOAD_PKL}" \
    --use_sc \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --motion_feat_dim ${MOTION_FEAT_DIM}

ok "Preloaded data pickle: ${DATA_PRELOAD_PKL}"

cd "${DITTO_TRAIN_DIR}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Training
# Only LMDM diffusion model is trained, everything else is frozen/offline
# ══════════════════════════════════════════════════════════════════════════════
log "PHASE 4: Starting training..."
log "  Experiment: ${EXPERIMENT_NAME}"
log "  GPUs: ${NUM_GPUS}, Batch/GPU: ${BATCH_SIZE}, Epochs: ${EPOCHS}"

cd "${DITTO_TRAIN_DIR}"

if [ "$NUM_GPUS" -gt 1 ]; then
    log "Launching distributed training with accelerate..."

    accelerate launch \
        --num_processes "${NUM_GPUS}" \
        --mixed_precision no \
        train_bridge_ditto.py \
        --experiment_name "${EXPERIMENT_NAME}" \
        --data_list_json "${DATA_LIST_JSON}" \
        --data_preload \
        --data_preload_pkl "${DATA_PRELOAD_PKL}" \
        --audio_feat_dim "${AUDIO_FEAT_DIM}" \
        --motion_feat_dim "${MOTION_FEAT_DIM}" \
        --seq_frames 80 \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --save_ckpt_freq 50 \
        --num_workers 4 \
        --use_accelerate \
        --use_emo \
        --use_eye_open \
        --use_eye_ball \
        --use_sc \
        --use_last_frame
else
    python train_bridge_ditto.py \
        --experiment_name "${EXPERIMENT_NAME}" \
        --data_list_json "${DATA_LIST_JSON}" \
        --data_preload \
        --data_preload_pkl "${DATA_PRELOAD_PKL}" \
        --audio_feat_dim "${AUDIO_FEAT_DIM}" \
        --motion_feat_dim "${MOTION_FEAT_DIM}" \
        --seq_frames 80 \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --save_ckpt_freq 50 \
        --num_workers 4 \
        --use_emo \
        --use_eye_open \
        --use_eye_ball \
        --use_sc \
        --use_last_frame
fi

echo ""
echo -e "${BOLD}${GREEN}"
echo "═══════════════════════════════════════════════════════════════"
echo "   ✅  Training Pipeline Complete!"
echo ""
echo "   Time elapsed: ${SECONDS}s"
echo ""
echo "   Checkpoints: ${DITTO_TRAIN_DIR}/experiments/s2/${EXPERIMENT_NAME}/ckpts/"
echo "   Loss log:    ${DITTO_TRAIN_DIR}/experiments/s2/${EXPERIMENT_NAME}/loss.log"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${RESET}"
