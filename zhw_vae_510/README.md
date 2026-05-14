# `zhw_vae_510` —— CrossView4DVAE standalone 训练

只跑 VAE，不接 ctsd。所有产物（checkpoints / previews / 日志）都写在 `zhw_vae_510/runs/`。

## 文件清单

| 文件 | 作用 |
|---|---|
| `data.py` | 三种数据源：`SyntheticCylinderDataset`（合成圆柱）、`MultiViewVAEAdapter`（包装 nuScenes / nuPlan / Waymo / Argoverse 任何 `MotionDataset`）、`make_nuscenes_base()`（nuScenes 工厂函数） |
| `losses.py` | 重建（L1/L2/Huber）+ KL（带 warmup）+ 可选 LPIPS |
| `train_vae.py` | 主训练脚本 |
| `overfit_sanity.py` | 5 分钟单 batch overfit 验证 |

---

## 推荐流程

### Linux 服务器环境准备

下面这组命令适合你现在这种 Linux 服务器 + `virtualenv` 环境。每次新开终端都先做：

```bash
cd /shareNFS_40/zhw/Per-step-ARDWM
source .venv/bin/activate
```

首次配环境时建议依次执行：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install pillow fsspec numpy tqdm einops opencv-python pyquaternion nuscenes-devkit
```

如果后面要开感知损失，再补：

```bash
python -m pip install lpips
```

然后确认 GPU / NCCL 正常：

```bash
python - <<'PY'
import torch
import torch.distributed as dist
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("nccl available:", dist.is_nccl_available())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
PY
```

### Linux 训练命令流程

建议严格按下面顺序跑，前一步通过再进下一步：

1. `py_compile` 检查脚本没有语法错误
2. 最小 synthetic smoke test，确认 train loop / ckpt / preview 可运行
3. `overfit_sanity.py` 验证模型 wiring
4. synthetic 完整训练
5. 两卡 DDP smoke test
6. 再切真实数据训练

对应命令如下。

#### 0. 语法检查

```bash
python -m py_compile \
    zhw_vae_510/train_vae.py \
    zhw_vae_510/data.py \
    zhw_vae_510/crossview_vae.py \
    zhw_vae_510/losses.py \
    zhw_vae_510/overfit_sanity.py
```

#### 1. 最小 single-GPU smoke test

```bash
python -m zhw_vae_510.train_vae \
    --data synthetic \
    --steps 2 \
    --image-hw 16 32 \
    --view-count 4 \
    --base-channels 8 \
    --batch-size 1 \
    --num-workers 0 \
    --amp none \
    --log-every 1 \
    --preview-every 2 \
    --ckpt-every 2
```

### Step 1：合成数据 sanity（5 分钟，保证模型通）

```bash
python -m zhw_vae_510.overfit_sanity --steps 300 --lr 5e-4 --base-channels 32 --image-hw 32 64 --view-count 4
```

期望终端打印 PSNR 单调上升（11 → 20+ dB），结束时显示 `>> OK` 或 `>> PASS`，并在 `zhw_vae_510/runs/overfit/overfit_final.png` 看到重建图。

> CPU 跑 300 步约需 3 分钟。GPU 上同样命令几十秒即可，建议加 `--steps 500` 让 PSNR 跨过 28 dB 拿到 PASS。

### Step 2：合成数据完整训练（确认管线 work）

```bash
python -m zhw_vae_510.train_vae \
    --data synthetic \
    --steps 5000 --image-hw 64 128 --base-channels 32 \
    --ckpt-every 200 --preview-every 200
```

#### DDP / 多卡训练

训练脚本支持 `torchrun` 单机多进程 DDP。`--batch-size` 是每张卡/每个进程的 batch size，全局 batch size = `--batch-size * nproc_per_node`。

先做一个 2 卡 smoke test：

```bash
torchrun --standalone --nproc_per_node=2 -m zhw_vae_510.train_vae \
    --data synthetic \
    --steps 2 \
    --image-hw 16 32 \
    --view-count 4 \
    --base-channels 8 \
    --batch-size 1 \
    --num-workers 0 \
    --amp none \
    --log-every 1 \
    --preview-every 2 \
    --ckpt-every 2
```

通过后再上正式多卡 synthetic：

```bash
torchrun --standalone --nproc_per_node=4 -m zhw_vae_510.train_vae --data synthetic \
    --steps 5000 --image-hw 64 128 --base-channels 32 \
    --batch-size 2 --num-workers 2
```

Windows 本机多进程通常使用 `gloo`；Linux CUDA 默认使用 `nccl`。如需手动指定：

```powershell
torchrun --standalone --nproc_per_node=2 -m zhw_vae_510.train_vae `
    --data synthetic `
    --steps 100 --image-hw 64 128 --base-channels 32 `
    --batch-size 1 --ddp-backend gloo
```

### Step 3：nuScenes 训练

参照 `configs/debug/rolling_ref+cleanN.json` 的路径约定，nuScenes 数据根目录里应该有：

```
<nusc_root>/
├── v1.0-trainval/                  # 元数据 json（dataset_name）
│   ├── sample_data.json
│   ├── sample.json
│   ├── ego_pose.json
│   └── ...
├── samples/                        # 关键帧 jpg
├── sweeps/                         # 非关键帧 jpg
└── (可选) maps/expansion/*.json    # 仅 hdmap 才需要
```

如果有预生成的 image cache：`<nusc_cache_root>/`（比 zip / 原图加载快）。

#### 命令示例（Linux 集群路径，按你 README 里的）

```bash
torchrun --standalone --nproc_per_node=4 -m zhw_vae_510.train_vae --data nuscenes \
    --nusc-data-root /inspire/hdd/.../dataset/nuscenes \
    --nusc-cache-root /inspire/hdd/.../dataset/nus_cache \
    --nusc-dataset-name v1.0-trainval \
    --nusc-split train \
    --nusc-fps 2 --nusc-stride 0.5 \
    --sequence-length 5 --image-hw 256 448 \
    --batch-size 1 --num-workers 2 \
    --base-channels 64 --steps 50000 \
    --use-camera-params
```

如果先单卡 smoke 一下，把前面的 `torchrun --standalone --nproc_per_node=4` 改成 `python -m`，并把 `--steps` 改成 `2` 或 `20`。

#### 命令示例（Windows）

```powershell
python -m zhw_vae_510.train_vae --data nuscenes `
    --nusc-data-root D:\path\to\nuscenes `
    --nusc-cache-root D:\path\to\nus_cache `
    --nusc-dataset-name v1.0-mini `
    --nusc-split mini_train `
    --sequence-length 5 --image-hw 256 448 `
    --batch-size 1 --num-workers 0 `
    --base-channels 32 --steps 5000 `
    --use-camera-params
```

> 用 `v1.0-mini` + `mini_train` 在本机先 smoke test，再切换到 `v1.0-trainval` + `train`。

---

## 关键参数说明

| 参数 | 含义 | 默认 | 说明 |
|---|---|---|---|
| `--sequence-length` | 时间帧数 T | 5 | 必须满足 `T = temporal_pre + k * tdf`；默认 `1 + k*4`，即 `1, 5, 9, 13` |
| `--image-hw` | 输入图像 resize 到 (H, W) | `64 128` | nuScenes 推荐 `256 448` |
| `--virtual-view-count` | 虚拟相机数（输出视角数） | -1（自动=V_in） | 自动匹配输入视角数；除非自己预先把 GT 投影到 V_virtual，否则不要手动改 |
| `--latent-view-count` | latent 视角数 | 3 | 3 个 120° 扇区覆盖 360°，比 2 (每扇区 180°) 的横向分辨率好近一倍。必须 ≤ virtual_view_count |
| `--base-channels` | 主干宽度 | 32 | 主要影响显存。24 GB 卡建议 32~64 |
| `--cylinder-radii` | 投影圆柱深度 (米) | `10.0` | 自动驾驶建议 `--cylinder-radii 3 10 30` 多深度聚合 |
| `--use-camera-params` | 启用几何投影 | off | nuScenes / nuPlan 建议开。否则走 fallback（最近相机分配） |
| `--kl-weight` | KL 权重 | 1e-6 | SD VAE 风格小值；KL 太大会塌陷 |
| `--kl-warmup` | KL 热身步数 | 2000 | 前期纯重建，让模型先学到东西再加 KL |
| `--amp` | 混合精度 | bf16 | A100/H100 用 bf16；老卡用 fp16 |
| `--ddp-backend` | DDP 后端 | 自动 | Linux CUDA 自动 `nccl`，Windows/CPU 自动 `gloo`；也可手动传 `gloo` |
| `--find-unused-parameters` | DDP unused 参数检查 | off | 只有遇到 DDP unused parameter 报错时再打开，会略慢 |
| `--perceptual-weight` | LPIPS 权重 | 0 | 设 >0 需 `pip install lpips` |

---

## 输出位置

```
zhw_vae_510/runs/
├── args.json                       # 本次训练的所有命令行参数
├── previews/
│   └── step_NNNNNN.png            # GT + 重建 side-by-side
└── ckpts/
    ├── step_NNNNNN.pt
    └── last.pt
```

恢复训练：`python -m zhw_vae_510.train_vae ... --resume zhw_vae_510/runs/ckpts/last.pt`

DDP 恢复训练：

```bash
torchrun --nproc_per_node=4 -m zhw_vae_510.train_vae ... \
    --resume zhw_vae_510/runs/ckpts/last.pt
```

DDP 下只有 rank 0 会写 `args.json`、preview 和 checkpoint；loss 日志会先在各 rank 间求平均。

---

## 常见问题

- **`AssertionError: T % stride == pre`**：`--sequence-length` 不满足 `T = temporal_pre + k * temporal_downsample_factor`。默认 `temporal_pre=1, tdf=4`，所以合法的 T 是 `1, 5, 9, 13, 17, ...`
- **`RuntimeError: expected mat1 and mat2 to have the same dtype`**：常见于 `--amp bf16` 时保存 preview。这个问题已经在当前版本修掉；如果你拉的是旧代码，更新 `train_vae.py` 后再跑。
- **`grid_sample 返回全 0` / 重建一直黑屏**：投影时内参缩放不对。我们已经修过这个 bug，会自动用 `image_size` 字段或 `cx*2` 反推标定分辨率，但前提是数据集要返回这些字段。如果仍出错，临时去掉 `--use-camera-params` 走 fallback。
- **OOM**：先降 `--base-channels 16` 和 `--image-hw 128 256`，或者降 `--sequence-length 5`，再逐步加大。
- **PSNR 上不去 / KL 塌陷**：先把 `--kl-weight 0` overfit 单 batch 看模型能不能学，能学就把 KL 慢慢加回来。
