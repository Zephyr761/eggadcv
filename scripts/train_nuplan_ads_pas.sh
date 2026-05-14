#!/usr/bin/env bash
set -e
set -x
trap 'echo "[FATAL] line=$LINENO cmd=$BASH_COMMAND" >&2' ERR

export NCCL_IB_TIMEOUT=22
export PYTORCH_NVML_BASED_CUDA_CHECK=1

# ===================== 环境变量检查与兼容处理 =====================
: "${NODE_RANK:=${RANK:-0}}"
: "${NNODES:=${WORLD_SIZE:-1}}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
export NODE_RANK NNODES MASTER_ADDR MASTER_PORT

echo "=== 分布式训练环境变量 ==="
echo "NODE_RANK=${NODE_RANK}, NNODES=${NNODES}, MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
echo "OUTPUT_URL=${OUTPUT_URL:-未设置}"

OBS_ROOT="obs://yw-2030-gy/external/personal/swx1481336"
CACHE_ROOT="/cache/swx1481336"
mkdir -p "${CACHE_ROOT}"

# ===================== 找 repo =====================
REPO_DIR=""
for d in /home/ma-user/* /home/ma-user/*/*; do
  [ -d "$d" ] || continue
  if [ -d "$d/configs" ] && [ -f "$d/src/dwm/train.py" ]; then
    REPO_DIR="$d"
    break
  fi
done
[ -z "$REPO_DIR" ] && { echo "cannot find repo" >&2; exit 3; }
echo "REPO_DIR=${REPO_DIR}"

# ===================== 下载工具 python =====================
MOX_PY="python"
"$MOX_PY" -c "import moxing as mox; import sys; print('MOX_PY=', sys.executable)" >/dev/null

# ===================== env 解压 =====================
ENV_DIR="${CACHE_ROOT}/envs/pas_env"
ENV_TAR_LOCAL="${CACHE_ROOT}/pas_env.tar.gz"

echo "=== [${NODE_RANK}] download & extract env ==="
"$MOX_PY" -c "import moxing as mox; mox.file.copy('${OBS_ROOT}/pas_env.tar.gz', '${ENV_TAR_LOCAL}')"
mkdir -p "${ENV_DIR}"
tar -xzf "${ENV_TAR_LOCAL}" -C "${ENV_DIR}" --checkpoint=1000 --checkpoint-action=dot
echo ""
sync

PY_BIN="${ENV_DIR}/bin/python"
[ ! -x "$PY_BIN" ] && for p in "${ENV_DIR}"/*/bin/python; do [ -x "$p" ] && { PY_BIN="$p"; break; }; done
[ ! -x "$PY_BIN" ] && { echo "cannot find python in ${ENV_DIR}" >&2; exit 4; }

echo "PY_BIN=$PY_BIN"
"$PY_BIN" -V
"$PY_BIN" -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())"
"$PY_BIN" -m pip install zstandard

# ===================== nuplan-devkit =====================
echo "=== [${NODE_RANK}] download & install nuplan-devkit ==="
NUPLAN_SRC_DIR="${CACHE_ROOT}/nuplan-devkit-master"
"$MOX_PY" -c "import moxing as mox; mox.file.copy_parallel('${OBS_ROOT}/nuplan-devkit-master/', '${NUPLAN_SRC_DIR}/')"
[ ! -f "${NUPLAN_SRC_DIR}/setup.py" ] && { echo "setup.py not found" >&2; exit 6; }
cd "${NUPLAN_SRC_DIR}" && "$PY_BIN" -m pip install -e . --no-deps && cd -
"$PY_BIN" -c "import nuplan; print('nuplan installed at:', nuplan.__file__)"

# ===================== 下载数据 =====================
echo "=== [${NODE_RANK}] download nuplan data ==="
NUCACHE_LOCAL="${CACHE_ROOT}/data/nuplan/nuplan_cache"
mkdir -p "${NUCACHE_LOCAL}"

"$MOX_PY" -c "import moxing as mox; mox.file.copy('${OBS_ROOT}/data/nuplan/nuplan_cache/mini_infos_train_pas.pkl', '${NUCACHE_LOCAL}/mini_infos_train_pas.pkl')"
"$MOX_PY" -c "import moxing as mox; mox.file.copy('${OBS_ROOT}/data/nuplan/nuplan_cache/mini_infos_val_pas.pkl', '${NUCACHE_LOCAL}/mini_infos_val_pas.pkl')"

# ===================== 通用 tar.zst 解压函数 =====================
extract_tar_zst() {
    local TAR_SRC="$1"
    local TAR_LOCAL="$2"
    local TARGET_DIR="$3"
    local DIR_NAME="$4"
    
    echo "=== Downloading ${DIR_NAME} ==="
    "$MOX_PY" -c "import moxing as mox; mox.file.copy('${TAR_SRC}', '${TAR_LOCAL}')"
    [ ! -f "${TAR_LOCAL}" ] && { echo "[ERROR] Download failed: ${TAR_LOCAL}" >&2; exit 1; }
    echo "Downloaded: $(du -h "${TAR_LOCAL}" | cut -f1)"
    
    rm -rf "${TARGET_DIR}"
    mkdir -p "${TARGET_DIR}"
    
    echo "=== Extracting ${DIR_NAME} ==="
    TAR_PATH="${TAR_LOCAL}" TARGET_PATH="${TARGET_DIR}" DIR_PREFIX="${DIR_NAME}" "$PY_BIN" - <<'PYSCRIPT'
import os, tarfile, zstandard as zstd, tempfile, time
tar_path = os.environ['TAR_PATH']
target_dir = os.environ['TARGET_PATH']
dir_prefix = os.environ['DIR_PREFIX']
base_dir = os.path.dirname(target_dir)

print(f"Extracting {tar_path} ({os.path.getsize(tar_path)/(1024**3):.2f} GB)")
start = time.time()

with open(tar_path, 'rb') as f:
    dctx = zstd.ZstdDecompressor()
    with tempfile.NamedTemporaryFile(suffix='.tar', delete=False, dir=base_dir) as tmp:
        tmp_path = tmp.name
        with dctx.stream_reader(f, read_size=16*1024*1024) as reader:
            while chunk := reader.read(16*1024*1024):
                tmp.write(chunk)

print(f"Decompressed in {time.time()-start:.1f}s, tar size: {os.path.getsize(tmp_path)/(1024**3):.2f}GB")

try:
    with tarfile.open(tmp_path, 'r') as tf:
        members = tf.getmembers()
        if not members:
            print("Error: Empty tar"); exit(1)
        extract_to = base_dir if members[0].name.startswith(dir_prefix+'/') else target_dir
        print(f"Extracting {len(members)} files to {extract_to}")
        for i, m in enumerate(members):
            tf.extract(m, extract_to)
            if (i+1) % 1000 == 0 or i == len(members)-1:
                print(f"{i+1}/{len(members)} ({(i+1)*100/len(members):.0f}%)", end='\r')
        print()
finally:
    os.path.exists(tmp_path) and os.unlink(tmp_path)

print(f"[SUCCESS] Extracted in {time.time()-start:.1f}s")
PYSCRIPT
    
    sync
    [ ! -d "${TARGET_DIR}" ] && { echo "[ERROR] ${TARGET_DIR} not created" >&2; exit 1; }
    echo "Verified: ${DIR_NAME} exists"
}

# 解压 LO.tar.zst
extract_tar_zst \
    "${OBS_ROOT}/LO.tar.zst" \
    "${NUCACHE_LOCAL}/LO.tar.zst" \
    "${NUCACHE_LOCAL}/LO_cache" \
    "LO_cache"

# 解压 pts_LO_ori.tar.zst
extract_tar_zst \
    "${OBS_ROOT}/cache/pts_LO_ori.tar.zst" \
    "${NUCACHE_LOCAL}/pts_LO_ori.tar.zst" \
    "${NUCACHE_LOCAL}/pts_LO_ori" \
    "pts_LO_ori"

# ===================== 下载其他数据 =====================
"$MOX_PY" -c "import moxing as mox; mox.file.copy_parallel('${OBS_ROOT}/data/nuplan/nuplan-v1.1/splits/', '${CACHE_ROOT}/data/nuplan/nuplan-v1.1/splits/')"
"$MOX_PY" -c "import moxing as mox; mox.file.copy_parallel('${OBS_ROOT}/data/nuplan/nuplan-v1.1/maps/', '${CACHE_ROOT}/data/nuplan/nuplan-v1.1/maps/')"

echo "=== [${NODE_RANK}] download sensor_blobs (MINI: 1 scene) ==="
SENSOR_DST="${CACHE_ROOT}/data/nuplan/nuplan-v1.1/sensor_blobs"

SCENE="2021.05.25.14.16.10_veh-35_01690_02183"
mkdir -p "${SENSOR_DST}"
"$MOX_PY" -c "import moxing as mox; mox.file.copy_parallel('${OBS_ROOT}/data/nuplan/nuplan-v1.1/sensor_blobs/${SCENE}/', '${SENSOR_DST}/${SCENE}/')"

echo "=== [${NODE_RANK}] download pretrain & cache ==="
"$MOX_PY" -c "import moxing as mox; mox.file.copy_parallel('${OBS_ROOT}/pretrain/', '${CACHE_ROOT}/pretrain/')"

CACHE_DATA_ROOT="${CACHE_ROOT}/data/cache"
mkdir -p "${CACHE_DATA_ROOT}"
"$MOX_PY" -c "import moxing as mox; mox.file.copy_parallel('${OBS_ROOT}/cache/actors/', '${CACHE_DATA_ROOT}/actors/')"
"$MOX_PY" -c "import moxing as mox; mox.file.copy_parallel('${OBS_ROOT}/cache/bg_out/', '${CACHE_DATA_ROOT}/bg_out/')"

# ===================== 多节点同步 =====================
echo "=== [${NODE_RANK}] waiting for all nodes ==="
READY_DIR=/data/log/download_ready
mkdir -p "$READY_DIR" && touch "$READY_DIR/rank${NODE_RANK}.ready"

WAIT_TIME=0
while [ "$(ls "$READY_DIR" | wc -l)" -lt "${NNODES:-1}" ]; do
    "$PY_BIN" -c "import time; time.sleep(10)"
    WAIT_TIME=$((WAIT_TIME+10))
    [ "$WAIT_TIME" -ge 7200 ] && { echo "Timeout waiting for nodes" >&2; exit 1; }
done
echo "All nodes ready, proceeding..."

# ===================== 下载 external/TAT =====================
echo "=== [${NODE_RANK}] download external/TAT ==="
mkdir -p "${CACHE_ROOT}/external/TAT/tats"
"$MOX_PY" -c "import moxing as mox; mox.file.copy_parallel('${OBS_ROOT}/external/TAT/tats/fvd/', '${CACHE_ROOT}/external/TAT/tats/fvd/')"

# ===================== 创建软链 =====================
USER_WORK="/home/ma-user/work/sWX1481336"
mkdir -p "${USER_WORK}/data/cache" "${USER_WORK}/data/nuplan" "${USER_WORK}/pretrain_ckpts"

ln -sfn "${CACHE_ROOT}/data/nuplan/nuplan_cache" "${USER_WORK}/data/cache/nuplan_cache"
ln -sfn "${CACHE_ROOT}/data/nuplan/nuplan-v1.1" "${USER_WORK}/data/nuplan/nuplan-v1.1"
ln -sfn "${CACHE_ROOT}/pretrain/nuplan_text.json" "${USER_WORK}/pretrain_ckpts/nuplan_text.json"
ln -sfn "${CACHE_ROOT}/pretrain/stabilityai" "${USER_WORK}/pretrain_ckpts/stabilityai"
ln -sfn "${CACHE_ROOT}/pretrain/opendwm-models" "${USER_WORK}/pretrain_ckpts/dwm"
ln -sfn "${CACHE_ROOT}/pretrain/14000.pth" "${USER_WORK}/pretrain_ckpts/14000.pth"
ln -sfn "${CACHE_ROOT}/data/cache/actors" "${USER_WORK}/data/cache/actors"
ln -sfn "${CACHE_ROOT}/data/cache/bg_out" "${USER_WORK}/data/cache/bg_out"
ln -sfn "${REPO_DIR}" "${USER_WORK}/Per-step-ARDWM-pas" || true

mkdir -p "${REPO_DIR}/externals"
ln -sfn "${CACHE_ROOT}/external/TAT" "${REPO_DIR}/externals/TAT"

# ===================== 训练环境配置 =====================
cd "${REPO_DIR}"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

NPROC_PER_NODE=8
[ -n "${CUDA_VISIBLE_DEVICES}" ] && NPROC_PER_NODE=$("$PY_BIN" -c "import os; s=os.environ.get('CUDA_VISIBLE_DEVICES','').strip(); print(len([x for x in s.split(',') if x.strip()]) if s else 8)")
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} => NPROC_PER_NODE=${NPROC_PER_NODE}"

export PYTHONPATH="${REPO_DIR}/src:${CACHE_ROOT}/external/TAT/tats:${CACHE_ROOT}/external/TAT/tats/fvd:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "=== verify pytorch_i3d ==="
"$PY_BIN" -c "import pytorch_i3d; print('pytorch_i3d OK:', pytorch_i3d.__file__)"

# ===================== 配置文件修改 =====================
: "${TRAIN_EPOCHS:=30}"
CFG="${REPO_DIR}/configs/debug/nuplan-train_224_pas_aug.json"
[ ! -f "$CFG" ] && { echo "missing cfg: $CFG" >&2; exit 5; }

export CFG TRAIN_EPOCHS NNODES NPROC_PER_NODE
"$PY_BIN" - <<'PY'
import json, os
cfg_path = os.environ["CFG"]
nnodes = int(os.environ.get("NNODES", "1"))
gpn = int(os.environ.get("NPROC_PER_NODE", "8"))
train_epochs = int(os.environ.get("TRAIN_EPOCHS", "10"))

with open(cfg_path) as f:
    d = json.load(f)

d.setdefault("global_state", {}).setdefault("device_mesh", {})["mesh_shape"] = [nnodes, gpn]
d["train_epochs"] = train_epochs

m = d.get("pipeline", {}).get("metrics", {})
if isinstance(m, dict):
    m.pop("fid", None) and print("removed fid")
    m.pop("fvd", None) and print("removed fvd")

with open(cfg_path, "w") as f:
    json.dump(d, f, indent=4)

print(f"mesh_shape={d['global_state']['device_mesh']['mesh_shape']}, epochs={d['train_epochs']}")
PY

# ===================== 启动训练 =====================
echo "=== 开始训练 (MINI版本：仅1个场景) ==="
echo "配置: ${CFG}, 节点: ${NNODES}, 当前: ${NODE_RANK}, GPUs: ${NPROC_PER_NODE}"

exec "$PY_BIN" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  src/dwm/train_adapt_data.py \
    --log-steps 500 --preview-steps 500 --checkpointing-steps 1000 --evaluation-steps 99999 \
    -c "${CFG}" -o "${OUTPUT_URL}"
