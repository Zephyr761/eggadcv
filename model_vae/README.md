# zhw_vae_510: Cross-view 4D VAE

`zhw_vae_510` 是一个独立的 cross-view 4D VAE 训练目录，只训练 VAE，不接 CTSD 主视频生成链路。默认输入是多帧多视角连续视频：

```text
vae_images: [B, T, V, 3, H, W]
range: [-1, 1]
```

其中 `T` 是连续时间帧，`V` 是相机视角数。你的 nus 数据如果已经输出 `[B,T,V,C,H,W]` 或单样本 `[T,V,C,H,W]` 图片 tensor，不需要额外特殊前处理；`MultiViewVAEAdapter` 会接受 `vae_images` tensor，并统一到 VAE 需要的 `[T,V,3,H,W]` / `[-1,1]`。如果底层 dataset 的 `__getitem__` 临时带了单样本 batch 维 `[1,T,V,C,H,W]`，adapter 也会自动去掉这个维度。

产物默认写到：

```text
zhw_vae_510/runs/
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `crossview_vae.py` | CrossView4DVAE 模型主体，包含 cylinder projector、causal 3D conv、view mixing、RoPE attention、view compression/expansion、Gaussian posterior。 |
| `data.py` | synthetic 数据、nuScenes/nus dataset factory、scene-folder loader、通用 `MultiViewVAEAdapter`、`vae_collate`。 |
| `losses.py` | VAE loss：重建 loss、KL warmup、可选 LPIPS、可选 logvar regularization。 |
| `train_vae.py` | 主训练入口，支持 single GPU / DDP、checkpoint、preview、metrics.jsonl。 |
| `overfit_sanity.py` | 单 batch deterministic overfit，用于先验证 wiring。 |
| `visual_debug.py` | 可视化 debug：输入、重建、activation、projector、RoPE 重合度。 |
| `plot_training.py` | 从 `metrics.jsonl` 生成 SVG 曲线或文本摘要。 |

## 模型架构

### 总体数据流

```text
x [B,T,V,3,H,W]
  -> stem causal 3D conv
  -> optional cylindrical view projector
  -> down1 / view_mix1
  -> down2 / view_mix2
  -> down3 / view_mix3
  -> pre_attn
  -> view_down
  -> attn
  -> to_moments -> Gaussian posterior(mean, logvar)
  -> z = posterior.mode() or posterior.sample()
  -> from_latent
  -> view_up
  -> post_attn
  -> up1 / view_mix_up1
  -> up2 / view_mix_up2
  -> up3
  -> head causal 3D conv
  -> recon [B,T,V,3,H,W]
```

### 输入与时间约束

模型输入输出统一为 `[B,T,V,C,H,W]`。训练时要求：

```text
T = temporal_pre + k * temporal_downsample_factor
```

默认：

```text
temporal_pre = 1
temporal_downsample_factor = 4
合法 T = 1, 5, 9, 13, ...
```

`train_vae.py` 和 `overfit_sanity.py` 会硬检查这个约束，避免 recon 和 target 静默按 `t_min` 截断。

### Stem

`stem` 是 `CausalConv3d(in_channels=3, out_channels=base_channels)`。每个 physical view 单独过 causal time conv：

```text
[B,T,V,3,H,W]
-> reshape [B*V,3,T,H,W]
-> causal 3D conv
-> [B,T,V,base_channels,H,W]
```

Causal padding 只向过去补帧，保证输出第 `t` 帧不看未来帧。

### CylindricalViewProjector

projector 有两条路径：

1. 无相机参数：fallback nearest-view path。默认 debug/overfit 推荐先用这条路径，不依赖 K/E。
2. 有相机参数：geometry-aware ray projection。给定 `camera_intrinsics`、`camera_transforms` 和 `intrinsics_hw` 后，把输入 feature 投到固定 virtual cylinder，再用 `grid_sample` 从源相机采样。

投影发生在 stem 后的 feature map 上，而不是 RGB 上，这样显存和计算更可控。`cylinder_radii` 默认 `[10.0]`，可以传多个半径做 multi-depth 聚合。

重要：真实 nus/nuScenes 训练建议先不加 `--use-camera-params`。当 fallback 路径能稳定下降后，再打开几何 projector；如果打开后 loss 不降，优先查 K/E、相机顺序、坐标系和 `intrinsics_hw`。

### Downsample Encoder

Encoder 有三组 time/space down block：

```text
down1: stride_t=1, stride_hw=2
down2: stride_t=2, stride_hw=2
down3: stride_t=max(1, temporal_downsample_factor // 2), stride_hw=2
```

默认 `temporal_downsample_factor=4` 时，时间总压缩约为 4，空间压缩为 8：

```text
T=5 -> latent T=2
H,W -> H/8,W/8
```

每个 `TimeSpaceDownBlock` 内部是：

```text
Conv3d stride downsample
GroupNorm per-frame
SiLU
CausalConv3d
GroupNorm per-frame
SiLU
```

GroupNorm 是 per-frame 做的，避免 3D norm 在时间维泄漏未来帧。

### ViewMixResBlock

`view_mix1/2/3` 和 decoder 侧 `view_mix_up1/2` 使用 `CircularConv3d` 在 view 轴做 circular padding。它的作用是让相邻 view 在圆环上混合：

```text
front-left <-> front
front-right <-> front
back-left <-> back
最后一个 view <-> 第一个 view
```

这一步只混 view/space，不混时间，所以仍保持 causal。

### RoPE3DBottleneckAttention

`pre_attn`、`attn`、`post_attn` 是 cross-view self-attention。它把每一帧独立展平成：

```text
[B,T,V,C,H,W] -> [B*T, V*H*W, C]
```

然后做 multi-head scaled dot-product attention。RoPE 坐标不是局部 view 坐标，而是全局 cylinder 坐标：

```text
azimuth: view + horizontal pixel 的全局环形位置
height: vertical pixel
```

这样不同 view 里看向同一物理方位的 token 会获得相近 RoPE phase，更容易对齐。

### ViewQueryCompressor

`view_down` 把 virtual views 压成 latent views：

```text
[B,T,V_virtual,C,H/8,W/8]
-> [B,T,V_latent,C,H/8,W/8]
```

它不是简单平均 view，而是 cross-attention：

```text
query: learnable latent-view query + input mean
key/value: virtual view tokens
RoPE: query/key 使用同一个 cylinder 坐标尺度
```

如果 overfit 不收敛，`view_down` 是重点检查对象之一。第一轮 sanity 建议：

```text
latent_view_count = view_count
latent_channels = 8 或 16
```

先证明不压 view 时能学，再逐渐加压缩。

### Gaussian Posterior

`to_moments` 是线性层：

```text
C_hidden -> 2 * latent_channels
```

输出拆成：

```text
mean, logvar
```

posterior 是 diagonal Gaussian：

```text
z = mean + eps * exp(0.5 * logvar)  # sample
z = mean                            # mode
```

训练入口默认使用 deterministic：

```text
sample_posterior = False
z = posterior.mode()
```

这是为了先排查 wiring。确认 deterministic overfit 能降后，再加 `--sample-posterior`、KL 和 logvar regularization。

### Decoder

Decoder 反向展开 latent：

```text
from_latent: latent_channels -> hidden C
view_up: V_latent -> V_virtual
post_attn: virtual-view self-attention
up1: time x temporal_downsample_factor//2, space x2
up2: time x2, space x2
up3: time x1, space x2
head: causal 3D conv -> RGB
```

`decode()` 会按：

```text
target_time = temporal_pre + (latent_time - 1) * temporal_downsample_factor
```

裁剪输出时间长度。因此输入 `T` 必须满足上面的时间约束。

### Loss

总 loss：

```text
loss = rec_weight * rec
     + perceptual_weight * LPIPS
     + kl_weight_now * KL
     + logvar_reg_weight * mean(logvar^2)
```

默认：

```text
rec_kind = l1
kl_weight = 1e-6
kl_warmup = 2000
perceptual_weight = 0
logvar_reg_weight = 0
```

日志里的 `kl` 是对 latent 元素求和后按 batch 平均，数值可能很大。看训练是否有效时优先看：

```text
rec
weighted_kl = kl * kl_weight_now
logvar_mean
preview image
```

## 数据格式与 nus 说明

训练主路径期望 batch 是：

```text
batch["vae_images"] = [B,T,V,3,H,W], range [-1,1]
```

`MultiViewVAEAdapter` 支持三类输入：

1. `sample["images"]`: nested PIL / tensor list `[T][V]`。
2. `sample["vae_images"]`: tensor `[T,V,C,H,W]` 或 `[T,V,H,W,C]`。
3. 单样本 batched tensor `[1,T,V,C,H,W]`，会自动 squeeze 成 `[T,V,C,H,W]`。

如果底层 nus dataset 已经输出连续视频 tensor，就不需要额外切帧、拼 view 或改 range；adapter 会处理 resize、channel、range。真正进入模型时会是 `[B,T,V,3,H,W]`。

相机参数可选：

```text
camera_intrinsics: [T,V,3,3] 或 [1,T,V,3,3]
camera_transforms: [T,V,4,4] 或 [1,T,V,4,4]
image_size: [T,V,2] 或 [1,T,V,2], 内容为 (W,H)
```

没有 K/E 也能训练，模型会走 fallback projector。

## Linux 服务器环境

原 README 中的工作目录路径保留如下：

```bash
cd /shareNFS_40/zhw/Per-step-ARDWM
source .venv/bin/activate
```

依赖安装：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install pillow fsspec numpy tqdm einops opencv-python pyquaternion nuscenes-devkit
```

如果要启用 perceptual loss：

```bash
python -m pip install lpips
```

检查 CUDA / NCCL：

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

## 训练前 smoke test

先做语法检查：

```bash
python -m py_compile \
  zhw_vae_510/train_vae.py \
  zhw_vae_510/data.py \
  zhw_vae_510/crossview_vae.py \
  zhw_vae_510/losses.py \
  zhw_vae_510/overfit_sanity.py \
  zhw_vae_510/visual_debug.py \
  zhw_vae_510/plot_training.py
```

最小 synthetic train smoke：

```bash
python -m zhw_vae_510.train_vae \
  --data synthetic \
  --steps 2 \
  --image-hw 16 32 \
  --sequence-length 5 \
  --view-count 4 \
  --latent-view-count 4 \
  --base-channels 8 \
  --latent-channels 8 \
  --batch-size 1 \
  --num-workers 0 \
  --amp none \
  --log-every 1 \
  --preview-every 2 \
  --ckpt-every 2 \
  --out zhw_vae_510/runs/smoke_synthetic
```

Deterministic overfit sanity：

```bash
python -m zhw_vae_510.overfit_sanity \
  --steps 500 \
  --lr 5e-4 \
  --image-hw 32 64 \
  --view-count 4 \
  --latent-view-count 4 \
  --latent-channels 8 \
  --base-channels 32 \
  --out zhw_vae_510/runs/overfit
```

如果 `rec_l1` 不降，先不要上真实 nus 数据。

## nus / nuScenes 训练命令

### 1. 你的 scene-folder nus 数据

原 README 里记录的 scene-folder 路径是：

```text
/shareNFS_40/sharedata/nuscenes/nuscenes_trainval
```

目录形态：

```text
<root>/
  train/
    scene-0001/
      images/
      opencv_cameras.json
    scene-0002/
      ...
  val/
    scene-xxxx/
      images/
      opencv_cameras.json
```

这种数据用：

```text
--data nuscenes_scene
```

单卡 smoke：

```bash
python -m zhw_vae_510.train_vae \
  --data nuscenes_scene \
  --nusc-data-root /shareNFS_40/sharedata/nuscenes/nuscenes_trainval \
  --nusc-split train \
  --sequence-length 5 \
  --image-hw 256 448 \
  --batch-size 1 \
  --num-workers 0 \
  --base-channels 32 \
  --latent-channels 8 \
  --latent-view-count 6 \
  --steps 20 \
  --log-every 1 \
  --preview-every 10 \
  --ckpt-every 20 \
  --amp bf16 \
  --out zhw_vae_510/runs/nusc_scene_smoke
```

正式单卡训练：

```bash
python -m zhw_vae_510.train_vae \
  --data nuscenes_scene \
  --nusc-data-root /shareNFS_40/sharedata/nuscenes/nuscenes_trainval \
  --nusc-split train \
  --sequence-length 5 \
  --image-hw 256 448 \
  --batch-size 1 \
  --num-workers 2 \
  --base-channels 64 \
  --latent-channels 8 \
  --latent-view-count 6 \
  --steps 50000 \
  --lr 2e-4 \
  --kl-weight 1e-6 \
  --kl-warmup 2000 \
  --log-every 10 \
  --preview-every 200 \
  --ckpt-every 500 \
  --amp bf16 \
  --out zhw_vae_510/runs/nusc_scene_vae_det
```

多卡 DDP：

```bash
torchrun --standalone --nproc_per_node=4 -m zhw_vae_510.train_vae \
  --data nuscenes_scene \
  --nusc-data-root /shareNFS_40/sharedata/nuscenes/nuscenes_trainval \
  --nusc-split train \
  --sequence-length 5 \
  --image-hw 256 448 \
  --batch-size 1 \
  --num-workers 2 \
  --base-channels 64 \
  --latent-channels 8 \
  --latent-view-count 6 \
  --steps 50000 \
  --lr 2e-4 \
  --kl-weight 1e-6 \
  --kl-warmup 2000 \
  --log-every 10 \
  --preview-every 200 \
  --ckpt-every 500 \
  --amp bf16 \
  --out zhw_vae_510/runs/nusc_scene_vae_ddp
```

建议第一轮不加 `--use-camera-params`。当 recon 能下降后，再对比：

```bash
python -m zhw_vae_510.train_vae \
  --data nuscenes_scene \
  --nusc-data-root /shareNFS_40/sharedata/nuscenes/nuscenes_trainval \
  --nusc-split train \
  --sequence-length 5 \
  --image-hw 256 448 \
  --batch-size 1 \
  --num-workers 2 \
  --base-channels 64 \
  --latent-channels 8 \
  --latent-view-count 6 \
  --steps 50000 \
  --use-camera-params \
  --amp bf16 \
  --out zhw_vae_510/runs/nusc_scene_vae_geom
```

如果加 `--use-camera-params` 后不降，先跑 visual debug 检查 projector。

### 2. 官方 nuScenes 表格式数据

原 README 里还保留了官方表格式路径占位：

```text
--nusc-data-root /inspire/hdd/.../dataset/nuscenes
--nusc-cache-root /inspire/hdd/.../dataset/nus_cache
```

官方表格式目录通常包含：

```text
<nusc_root>/
  v1.0-trainval/
    sample_data.json
    sample.json
    ego_pose.json
    ...
  samples/
  sweeps/
```

训练命令：

```bash
torchrun --standalone --nproc_per_node=4 -m zhw_vae_510.train_vae \
  --data nus \
  --nusc-data-root /inspire/hdd/.../dataset/nuscenes \
  --nusc-cache-root /inspire/hdd/.../dataset/nus_cache \
  --nusc-dataset-name v1.0-trainval \
  --nusc-split train \
  --nusc-fps 2 \
  --nusc-stride 0.5 \
  --sequence-length 5 \
  --image-hw 256 448 \
  --batch-size 1 \
  --num-workers 2 \
  --base-channels 64 \
  --latent-channels 8 \
  --latent-view-count 6 \
  --steps 50000 \
  --amp bf16 \
  --out zhw_vae_510/runs/nus_official_vae
```

`--data nus` 是 `--data nuscenes` 的别名。

### 3. 如果 nus dataset 已经直接输出视频 tensor

如果你自己的 nus dataset 已经返回：

```python
{
    "vae_images": tensor,  # [T,V,C,H,W] or [1,T,V,C,H,W]
}
```

或者甚至 `__getitem__` 直接返回 tensor，`MultiViewVAEAdapter` 会直接接收，不需要 PIL resize 逻辑。注意：

```text
dataset __getitem__ 最好返回单条 clip: [T,V,C,H,W]
如果临时有 [1,T,V,C,H,W]，adapter 会 squeeze
如果是 [B,T,V,C,H,W] 且 B > 1，需要在 dataset 内拆成单样本，否则 DataLoader 会再次 batch
```

## 采样、KL 与 logvar 建议

排查不收敛时按顺序来：

1. deterministic：默认不加 `--sample-posterior`。
2. 不压 view：`--latent-view-count` 等于输入 view 数。
3. 加容量：`--latent-channels 8` 或 `16`。
4. 确认 rec 明显下降后，再打开：

```bash
--sample-posterior --logvar-reg-weight 1e-4
```

如果 sampling 打开后 rec 抖动或不降，重点看 `logvar_mean` 和 `posterior_logvar` 的 visual debug。

## 可视化 debug

`visual_debug.py` 用一次 forward 生成图片、JSON 和 Markdown 报告。

### Synthetic fallback projector

```bash
python -m zhw_vae_510.visual_debug \
  --data synthetic \
  --image-hw 32 64 \
  --view-count 4 \
  --sequence-length 5 \
  --base-channels 16 \
  --latent-channels 8 \
  --out zhw_vae_510/runs/visual_debug_synth \
  --device cuda
```

### nus scene-folder

```bash
python -m zhw_vae_510.visual_debug \
  --data nuscenes_scene \
  --nusc-data-root /shareNFS_40/sharedata/nuscenes/nuscenes_trainval \
  --nusc-split train \
  --sample-index 0 \
  --sequence-length 5 \
  --image-hw 256 448 \
  --base-channels 32 \
  --latent-channels 8 \
  --latent-view-count 6 \
  --out zhw_vae_510/runs/visual_debug_nus \
  --device cuda
```

### 检查 checkpoint

```bash
python -m zhw_vae_510.visual_debug \
  --data nuscenes_scene \
  --nusc-data-root /shareNFS_40/sharedata/nuscenes/nuscenes_trainval \
  --nusc-split train \
  --checkpoint zhw_vae_510/runs/nusc_scene_vae_det/ckpts/last.pt \
  --sequence-length 5 \
  --image-hw 256 448 \
  --base-channels 64 \
  --latent-channels 8 \
  --latent-view-count 6 \
  --out zhw_vae_510/runs/visual_debug_ckpt \
  --device cuda
```

### 检查 geometry projector

```bash
python -m zhw_vae_510.visual_debug \
  --data nuscenes_scene \
  --nusc-data-root /shareNFS_40/sharedata/nuscenes/nuscenes_trainval \
  --nusc-split train \
  --use-camera-params \
  --sequence-length 5 \
  --image-hw 256 448 \
  --base-channels 32 \
  --latent-channels 8 \
  --latent-view-count 6 \
  --out zhw_vae_510/runs/visual_debug_geom \
  --device cuda
```

### 输出文件

```text
report.md
debug_metrics.json
input/input_grid.png
input/temporal_delta.png
reconstruction/recon_grid.png
reconstruction/abs_error.png
activations/*_energy.png
activations/*_view_cosine.png
projector/coverage.png
projector/source_entropy.png
projector/source_camera_rgb.png
rope/*_nearest_distance.png
rope/*_query_azimuth.png
```

### 怎么判断有没有 bug

#### 输入数据

看：

```text
input/input_grid.png
input/temporal_delta.png
debug_metrics.json -> input
```

异常信号：

- 某些 view 全黑或全灰。
- view 顺序明显错乱。
- synthetic 的 `temporal_delta` 像随机噪声一样全亮。
- `input.min/max` 不在 `[-1,1]` 附近。

#### 重建

看：

```text
reconstruction/recon_grid.png
reconstruction/abs_error.png
reconstruction_l1
```

判断：

- 全图 error 高：模型没学到或 posterior sampling/logvar 干扰。
- seam 附近 error 高：view order、circular padding、projector 或 RoPE 对齐问题。
- 运动区域 error 高：时间压缩太强或 temporal causal stack 容量不足。

#### Activation

看：

```text
activations/*_energy.png
activations/*_view_cosine.png
```

异常信号：

- 某阶段 energy 全黑：dead feature。
- 某阶段 `std` 突然极大：爆激活。
- 某个 view 一直低能量：某 view 数据或 projector 掉了。
- 很早的 stage view cosine 几乎全 1：view collapse。
- 到 `pre_attn/attn` 后 cosine 仍几乎只有对角线亮：跨 view 没有混起来。

#### Projector

看：

```text
projector/coverage.png
projector/source_entropy.png
projector/source_camera_rgb.png
debug_metrics.json -> projector
```

指标：

```text
coverage_zero_fraction 越低越好
coverage_mean 不应接近 0
entropy_mean fallback 通常接近 0
geometry path 有少量边缘空洞可以接受，但大面积空洞不正常
```

异常时优先查：

- K 是否对应 resize 前分辨率。
- `intrinsics_hw` 是否正确。
- `camera_transforms` 是 camera-to-ego 还是 ego-to-camera。
- view 顺序是否和图片顺序一致。

#### RoPE

看：

```text
rope/self_attention_nearest_distance.png
rope/view_compression_nearest_distance.png
rope/view_expansion_nearest_distance.png
debug_metrics.json -> rope
```

判断：

- `nearest_distance_p95` 通常应小于约 1 pseudo-pixel。
- `internal_seam_step_mean` 应接近 1。
- 如果 nearest distance 图有大条纹，说明 query/key cylinder 坐标没有对齐。
- 如果 compression/expansion 的 nearest distance 高，重点查 `latent_view_count` 和 `virtual_view_count` 的坐标尺度。

## 曲线查看

文本摘要：

```bash
python -m zhw_vae_510.plot_training \
  --run zhw_vae_510/runs/nusc_scene_vae_det \
  --text-only
```

生成 SVG：

```bash
python -m zhw_vae_510.plot_training \
  --run zhw_vae_510/runs/nusc_scene_vae_det
```

输出：

```text
zhw_vae_510/runs/nusc_scene_vae_det/training_curves.svg
```

## 常见排查顺序

1. `python -m py_compile ...`
2. synthetic smoke。
3. `overfit_sanity` deterministic，不压 view。
4. nus single batch / small step，不加 `--use-camera-params`。
5. `visual_debug` 看 input、activation、RoPE。
6. 如果 fallback 能降，再打开 `--use-camera-params`。
7. 如果 deterministic 能降，再打开 `--sample-posterior` 和 KL。

## 当前实现已检查过的跑通项

在本地已检查：

```text
ruff check zhw_vae_510: pass
py_compile core scripts: pass
python -m zhw_vae_510.data: pass
python -m zhw_vae_510.overfit_sanity --steps 1: pass
python -m zhw_vae_510.train_vae --steps 1 synthetic: pass
python -m zhw_vae_510.visual_debug synthetic: pass
python -m zhw_vae_510.plot_training --text-only: pass
```

Windows 本地如果遇到 OpenMP runtime 冲突，可临时：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

Linux 服务器一般不需要这个变量。
