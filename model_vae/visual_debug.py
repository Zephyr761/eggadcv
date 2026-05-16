"""Visual debugging for CrossView4DVAE.

This script runs one batch through the VAE and writes a compact debug report:

- input / reconstruction / absolute error grids;
- temporal input deltas;
- activation statistics and per-stage energy maps;
- per-stage cross-view cosine similarity matrices;
- projector source-camera coverage maps using synthetic one-hot camera IDs;
- RoPE coordinate overlap diagnostics for self-attention, view compression,
  and view expansion.

Everything is written under ``zhw_vae_510/runs`` by default.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zhw_vae_510.crossview_vae import (  # noqa: E402
    CrossView4DVAE,
    _build_view_coords,
)
from zhw_vae_510.data import (  # noqa: E402
    MultiViewVAEAdapter,
    SyntheticConfig,
    SyntheticCylinderDataset,
    make_nuscenes_base,
    make_nuscenes_scene_folder_base,
    vae_collate,
)


STAGE_NAMES = [
    "virtual_projector",
    "down1",
    "view_mix1",
    "down2",
    "view_mix2",
    "down3",
    "view_mix3",
    "pre_attn",
    "view_down",
    "attn",
    "to_moments",
    "from_latent",
    "view_up",
    "post_attn",
    "up1",
    "view_mix_up1",
    "up2",
    "view_mix_up2",
    "up3",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", choices=["synthetic", "nus", "nuscenes", "nuscenes_scene", "nuplan"],
                   default="synthetic")
    p.add_argument("--out", default="zhw_vae_510/runs/visual_debug")
    p.add_argument("--checkpoint", default="",
                   help="Optional train_vae checkpoint. Loads key 'model'.")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--device", default="cuda")

    p.add_argument("--image-hw", type=int, nargs=2, default=[64, 128])
    p.add_argument("--sequence-length", type=int, default=5)
    p.add_argument("--view-count", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--nusc-data-root", default="")
    p.add_argument("--nusc-cache-root", default="")
    p.add_argument("--nusc-dataset-name", default="v1.0-trainval")
    p.add_argument("--nusc-split", default="train")
    p.add_argument("--nusc-fps", type=int, default=2)
    p.add_argument("--nusc-stride", type=float, default=0.5)
    p.add_argument("--nusc-keyframe-only", action="store_true", default=True)

    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--latent-channels", type=int, default=8)
    p.add_argument("--virtual-view-count", type=int, default=-1)
    p.add_argument("--latent-view-count", type=int, default=-1,
                   help="Default -1 means match input view count for debugging.")
    p.add_argument("--num-attention-heads", type=int, default=2)
    p.add_argument("--num-bottleneck-blocks", type=int, default=1)
    p.add_argument("--temporal-downsample-factor", type=int, default=4)
    p.add_argument("--temporal-pre", type=int, default=1)
    p.add_argument("--spatial-downsample-factor", type=int, default=8)
    p.add_argument("--cylinder-radii", type=float, nargs="+", default=[10.0])

    p.add_argument("--use-camera-params", action="store_true",
                   help="Use K/E in encode and projector coverage diagnostics.")
    p.add_argument("--sample-posterior", action="store_true",
                   help="Use stochastic z. Default uses posterior.mode().")
    p.add_argument("--max-frames", type=int, default=5)
    p.add_argument("--max-views", type=int, default=8)
    return p


def validate_temporal_config(args: argparse.Namespace) -> None:
    delta = args.sequence_length - args.temporal_pre
    if delta < 0 or delta % args.temporal_downsample_factor != 0:
        raise ValueError(
            "Invalid sequence_length: expected "
            "T = temporal_pre + k * temporal_downsample_factor, got "
            f"T={args.sequence_length}, temporal_pre={args.temporal_pre}, "
            f"tdf={args.temporal_downsample_factor}."
        )


def make_dataset(args: argparse.Namespace):
    if args.data == "synthetic":
        return SyntheticCylinderDataset(SyntheticConfig(
            samples_per_epoch=max(args.sample_index + 1, 16),
            sequence_length=args.sequence_length,
            view_count=args.view_count,
            image_hw=tuple(args.image_hw),
            seed=args.seed,
        ))

    if args.data == "nuscenes_scene":
        if not args.nusc_data_root:
            raise ValueError("--nusc-data-root is required for nuscenes_scene")
        base = make_nuscenes_scene_folder_base(
            data_root=args.nusc_data_root,
            split=args.nusc_split,
            sequence_length=args.sequence_length,
        )
        return MultiViewVAEAdapter(
            base, sequence_length=args.sequence_length,
            image_hw=tuple(args.image_hw))

    if args.data in ("nus", "nuscenes"):
        if not args.nusc_data_root:
            raise ValueError("--nusc-data-root is required for nus/nuscenes")
        base = make_nuscenes_base(
            data_root=args.nusc_data_root,
            cache_root=args.nusc_cache_root or None,
            dataset_name=args.nusc_dataset_name,
            split=args.nusc_split,
            sequence_length=args.sequence_length,
            fps=args.nusc_fps,
            stride=args.nusc_stride,
            keyframe_only=args.nusc_keyframe_only,
        )
        return MultiViewVAEAdapter(
            base, sequence_length=args.sequence_length,
            image_hw=tuple(args.image_hw))

    from dwm.tools.dataset_nus import make_base_ds
    base = make_base_ds(train=True)
    return MultiViewVAEAdapter(
        base, sequence_length=args.sequence_length,
        image_hw=tuple(args.image_hw))


def load_checkpoint_if_needed(model: CrossView4DVAE, path: str) -> None:
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[checkpoint] loaded {path}; "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().float()
    finite = torch.isfinite(y)
    if not finite.any():
        return {
            "shape": list(x.shape),
            "finite_fraction": 0.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "zero_fraction": None,
        }
    yf = y[finite]
    return {
        "shape": list(x.shape),
        "finite_fraction": float(finite.float().mean().item()),
        "min": float(yf.min().item()),
        "max": float(yf.max().item()),
        "mean": float(yf.mean().item()),
        "std": float(yf.std(unbiased=False).item()),
        "zero_fraction": float((yf.abs() < 1e-8).float().mean().item()),
    }


def normalize01(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().float()
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    lo = y.min()
    hi = y.max()
    if (hi - lo).abs() < 1e-12:
        return torch.zeros_like(y)
    return (y - lo) / (hi - lo)


def heatmap_rgb(x: torch.Tensor) -> torch.Tensor:
    y = normalize01(x)
    r = y
    g = (1.0 - (2.0 * y - 1.0).abs()).clamp(0, 1)
    b = 1.0 - y
    return torch.stack([r, g, b], dim=0)


def matrix_heatmap(mat: torch.Tensor, cell: int = 28) -> torch.Tensor:
    m = heatmap_rgb(mat)
    return F.interpolate(
        m.unsqueeze(0), scale_factor=cell, mode="nearest").squeeze(0)


def save_image_grid(path: Path, images: torch.Tensor, nrow: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images.detach().cpu(), str(path), nrow=nrow)


def save_video_view_grid(path: Path, x: torch.Tensor, max_frames: int,
                         max_views: int) -> None:
    # x: [B,T,V,3,H,W] in [-1,1] or [0,1].
    y = x.detach().cpu()[0]
    t = min(y.shape[0], max_frames)
    v = min(y.shape[1], max_views)
    y = y[:t, :v]
    if y.min() < -0.05:
        y = (y.clamp(-1, 1) + 1.0) / 2.0
    y = y.clamp(0, 1).reshape(t * v, y.shape[2], y.shape[3], y.shape[4])
    save_image_grid(path, y, nrow=v)


def save_tensor_heat_grid(path: Path, maps: torch.Tensor, max_frames: int,
                          max_views: int) -> None:
    # maps: [T,V,H,W]
    t = min(maps.shape[0], max_frames)
    v = min(maps.shape[1], max_views)
    imgs = [heatmap_rgb(maps[i, j]) for i in range(t) for j in range(v)]
    save_image_grid(path, torch.stack(imgs), nrow=v)


def as_6d_activation(name: str, x: torch.Tensor) -> torch.Tensor | None:
    if x.ndim != 6:
        return None
    # Heuristic: [B,T,V,C,H,W] has channel dim before H/W; Linear hooks have
    # [B,T,V,H,W,C].
    if name in {"to_moments", "from_latent"}:
        return x.permute(0, 1, 2, 5, 3, 4).contiguous()
    return x


def view_cosine_matrix(act: torch.Tensor) -> torch.Tensor:
    # act: [B,T,V,C,H,W]. Average batch/time/spatial, compare view feature vecs.
    y = act.detach().float()
    y = y.mean(dim=(0, 1, 4, 5))  # [V,C]
    y = F.normalize(y, dim=-1, eps=1e-6)
    return y @ y.t()


def save_activation_debug(out_dir: Path, name: str, tensor: torch.Tensor,
                          max_frames: int, max_views: int) -> dict[str, Any]:
    stats = tensor_stats(tensor)
    act = as_6d_activation(name, tensor)
    if act is None:
        return stats

    energy = act.detach().float().abs().mean(dim=3)[0]  # [T,V,H,W]
    stats["per_time_view_energy"] = (
        energy.mean(dim=(-2, -1)).detach().cpu().tolist()
    )
    save_tensor_heat_grid(
        out_dir / "activations" / f"{name}_energy.png",
        energy,
        max_frames=max_frames,
        max_views=max_views,
    )
    cos = view_cosine_matrix(act)
    stats["view_cosine"] = cos.detach().cpu().tolist()
    save_image_grid(
        out_dir / "activations" / f"{name}_view_cosine.png",
        matrix_heatmap(cos).unsqueeze(0),
        nrow=1,
    )
    return stats


def camera_palette(n: int) -> torch.Tensor:
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
        colors.append([r, g, b])
    return torch.tensor(colors, dtype=torch.float32)


def projector_debug(model: CrossView4DVAE, batch: dict[str, Any],
                    use_camera_params: bool, out_dir: Path, max_frames: int,
                    max_views: int, device: torch.device) -> dict[str, Any]:
    x = batch["vae_images"].to(device)
    b, t, v, _c, h, w = x.shape
    ids = torch.eye(v, device=device).view(1, 1, v, v, 1, 1)
    ids = ids.expand(b, t, v, v, h, w).contiguous()

    K = batch.get("camera_intrinsics")
    E = batch.get("camera_transforms")
    intr_hw = batch.get("intrinsics_hw")
    if use_camera_params and K is not None and E is not None:
        K = K.to(device)
        E = E.to(device)
    else:
        K = None
        E = None

    with torch.no_grad():
        projected = model.virtual_projector(ids, K, E, intr_hw)

    # projected: [B,T,V_virtual,C_source,H,W].
    weights = projected[0].detach().float().clamp(min=0)
    coverage = weights.sum(dim=2)  # [T,V_virtual,H,W]
    probs = weights / coverage.unsqueeze(2).clamp(min=1e-8)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=2)
    if v > 1:
        entropy = entropy / math.log(v)

    save_tensor_heat_grid(
        out_dir / "projector" / "coverage.png",
        coverage,
        max_frames=max_frames,
        max_views=max_views,
    )
    save_tensor_heat_grid(
        out_dir / "projector" / "source_entropy.png",
        entropy,
        max_frames=max_frames,
        max_views=max_views,
    )

    palette = camera_palette(v).to(weights.device)
    rgb = torch.einsum("tvchw,cr->tvrhw", probs, palette)
    t_keep = min(rgb.shape[0], max_frames)
    v_keep = min(rgb.shape[1], max_views)
    rgb_imgs = rgb[:t_keep, :v_keep].reshape(
        t_keep * v_keep, 3, rgb.shape[-2], rgb.shape[-1])
    save_image_grid(out_dir / "projector" / "source_camera_rgb.png",
                    rgb_imgs, nrow=v_keep)

    dominant = probs.mean(dim=(-2, -1)).argmax(dim=-1)  # [T,V_virtual]
    stats = {
        "projector_used_camera_params": bool(K is not None and E is not None),
        "weights_shape": list(projected.shape),
        "coverage_min": float(coverage.min().item()),
        "coverage_mean": float(coverage.mean().item()),
        "coverage_zero_fraction": float((coverage < 1e-4).float().mean().item()),
        "entropy_mean": float(entropy.mean().item()),
        "entropy_p95": float(torch.quantile(entropy.flatten(), 0.95).item()),
        "dominant_source_by_time_virtual_view": dominant.cpu().tolist(),
    }
    return stats


def nearest_coord_distance(q_az: torch.Tensor, q_h: torch.Tensor,
                           k_az: torch.Tensor, k_h: torch.Tensor,
                           chunk: int = 4096) -> torch.Tensor:
    outs = []
    keys = torch.stack([k_az, k_h], dim=1)
    for start in range(0, q_az.numel(), chunk):
        q = torch.stack(
            [q_az[start:start + chunk], q_h[start:start + chunk]], dim=1)
        d = torch.cdist(q.float(), keys.float())
        outs.append(d.min(dim=1).values)
    return torch.cat(outs, dim=0)


def seam_stats(view_count: int, h: int, w: int) -> dict[str, float]:
    az, _hh = _build_view_coords(view_count, h, w, "cpu", torch.float32)
    az = az.view(view_count, h, w)
    internal = []
    for v in range(view_count - 1):
        internal.append((az[v + 1, :, 0] - az[v, :, -1]).mean())
    wrap = (az[0, :, 0] - az[-1, :, -1]).mean()
    return {
        "internal_seam_step_mean": float(torch.stack(internal).mean().item())
        if internal else 0.0,
        "wrap_seam_step": float(wrap.item()),
    }


def rope_pair_debug(out_dir: Path, label: str, q_views: int, k_views: int,
                    h: int, w: int, max_views: int) -> dict[str, Any]:
    ref = max(q_views, k_views)
    q_az, q_h = _build_view_coords(q_views, h, w, "cpu", torch.float32, ref)
    k_az, k_h = _build_view_coords(k_views, h, w, "cpu", torch.float32, ref)
    dist = nearest_coord_distance(q_az, q_h, k_az, k_h)
    maps = dist.view(q_views, h, w).unsqueeze(0)  # [T=1,V,H,W]
    save_tensor_heat_grid(
        out_dir / "rope" / f"{label}_nearest_distance.png",
        maps,
        max_frames=1,
        max_views=max_views,
    )
    q_az_map = q_az.view(q_views, h, w).unsqueeze(0)
    save_tensor_heat_grid(
        out_dir / "rope" / f"{label}_query_azimuth.png",
        q_az_map,
        max_frames=1,
        max_views=max_views,
    )
    return {
        "label": label,
        "query_views": q_views,
        "key_views": k_views,
        "height": h,
        "width": w,
        "nearest_distance_mean": float(dist.mean().item()),
        "nearest_distance_p95": float(torch.quantile(dist, 0.95).item()),
        "nearest_distance_max": float(dist.max().item()),
        "query_seams": seam_stats(q_views, h, w),
        "key_seams": seam_stats(k_views, h, w),
    }


def rope_debug(out_dir: Path, activations: dict[str, torch.Tensor],
               model: CrossView4DVAE, max_views: int) -> dict[str, Any]:
    stats: dict[str, Any] = {}

    pre = activations.get("view_mix3")
    if pre is not None:
        act = as_6d_activation("view_mix3", pre)
        if act is not None:
            _b, _t, v, _c, h, w = act.shape
            stats["self_attention"] = rope_pair_debug(
                out_dir, "self_attention", v, v, h, w, max_views)

    down = activations.get("view_down")
    if down is not None and pre is not None:
        pre_act = as_6d_activation("view_mix3", pre)
        down_act = as_6d_activation("view_down", down)
        if pre_act is not None and down_act is not None:
            _b, _t, k_views, _c, h, w = pre_act.shape
            q_views = down_act.shape[2]
            stats["view_compression"] = rope_pair_debug(
                out_dir, "view_compression", q_views, k_views, h, w, max_views)

    z_views = int(model.config.latent_view_count)
    v_views = int(model.config.virtual_view_count)
    latent_h = latent_w = None
    if down is not None:
        down_act = as_6d_activation("view_down", down)
        if down_act is not None:
            latent_h = down_act.shape[-2]
            latent_w = down_act.shape[-1]
    if latent_h is not None and latent_w is not None:
        stats["view_expansion"] = rope_pair_debug(
            out_dir, "view_expansion", v_views, z_views,
            latent_h, latent_w, max_views)
    return stats


def capture_activations(model: CrossView4DVAE) -> tuple[dict[str, torch.Tensor], list[Any]]:
    activations: dict[str, torch.Tensor] = {}
    handles = []
    modules = dict(model.named_modules())

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                activations[name] = output.detach().cpu()
        return hook

    for name in STAGE_NAMES:
        module = modules.get(name)
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(name)))
    return activations, handles


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_report(out_dir: Path, args: argparse.Namespace,
                 metrics: dict[str, Any]) -> None:
    lines = [
        "# CrossView4DVAE Visual Debug Report",
        "",
        "## What Was Visualized",
        "",
        "- `input/input_grid.png`: raw VAE input. Rows are time, columns are views.",
        "- `input/temporal_delta.png`: per-view L1 change between adjacent frames.",
        "- `reconstruction/recon_grid.png`: deterministic reconstruction.",
        "- `reconstruction/abs_error.png`: absolute reconstruction error.",
        "- `activations/*_energy.png`: mean absolute feature energy by stage.",
        "- `activations/*_view_cosine.png`: cross-view feature cosine matrix.",
        "- `projector/coverage.png`: how much source-camera signal reaches each virtual pixel.",
        "- `projector/source_entropy.png`: whether projector blends many cameras or picks one.",
        "- `projector/source_camera_rgb.png`: dominant source-camera color map.",
        "- `rope/*_nearest_distance.png`: nearest RoPE query/key coordinate distance.",
        "- `rope/*_query_azimuth.png`: query azimuth coordinate layout.",
        "",
        "## How To Read It",
        "",
        "- Input bug: blank/dark columns in `input_grid.png`, view means near zero, or huge random `temporal_delta` on synthetic data.",
        "- Projector bug: `coverage_zero_fraction` high, all-black coverage, or source colors not matching expected camera order.",
        "- RoPE bug: internal seam step should be close to 1. Nearest-distance p95 should usually be below about 1 pseudo-pixel. Large stripes mean query/key coordinates are misaligned.",
        "- View collapse: stage `view_cosine` matrices become almost all 1.0 very early, or one view has near-zero energy across stages.",
        "- Dead/exploding stage: activation `std` near 0, `zero_fraction` near 1, non-finite values, or a sudden 10x energy jump between adjacent stages.",
        "- Bottleneck too strong: activations are healthy but reconstruction error remains structured and high, especially after `view_down`.",
        "",
        "## Run Args",
        "",
        "```json",
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Key Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    validate_temporal_config(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.")

    dataset = make_dataset(args)
    sample = dataset[args.sample_index]
    batch = vae_collate([sample])
    x = batch["vae_images"].to(device)
    b, t, v_input, _c, h, w = x.shape

    virtual_views = args.virtual_view_count if args.virtual_view_count > 0 else v_input
    if virtual_views != v_input:
        raise ValueError(
            "visual_debug expects virtual_view_count == input view count so "
            f"reconstruction can be compared directly, got {virtual_views} vs {v_input}."
        )
    latent_views = args.latent_view_count if args.latent_view_count > 0 else v_input
    if latent_views > virtual_views:
        raise ValueError("latent_view_count must be <= virtual_view_count")

    model = CrossView4DVAE(
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        virtual_view_count=virtual_views,
        latent_view_count=latent_views,
        num_attention_heads=args.num_attention_heads,
        num_bottleneck_blocks=args.num_bottleneck_blocks,
        temporal_downsample_factor=args.temporal_downsample_factor,
        temporal_pre=args.temporal_pre,
        spatial_downsample_factor=args.spatial_downsample_factor,
        cylinder_radii=tuple(args.cylinder_radii),
    ).to(device)
    load_checkpoint_if_needed(model, args.checkpoint)
    model.eval()

    K = batch.get("camera_intrinsics")
    E = batch.get("camera_transforms")
    intr_hw = batch.get("intrinsics_hw")
    if args.use_camera_params and K is not None and E is not None:
        K = K.to(device)
        E = E.to(device)
    else:
        K = None
        E = None

    save_video_view_grid(
        out_dir / "input" / "input_grid.png",
        x.detach().cpu(),
        max_frames=args.max_frames,
        max_views=args.max_views,
    )
    if t > 1:
        delta = (x[:, 1:] - x[:, :-1]).abs().mean(dim=3)[0].detach().cpu()
        save_tensor_heat_grid(
            out_dir / "input" / "temporal_delta.png",
            delta,
            max_frames=args.max_frames,
            max_views=args.max_views,
        )

    activations, handles = capture_activations(model)
    with torch.no_grad():
        output = model(
            x,
            sample_posterior=args.sample_posterior,
            intrinsics=K,
            extrinsics=E,
            intrinsics_hw=intr_hw,
        )
    for handle in handles:
        handle.remove()

    recon = output["sample"]
    if recon.shape != x.shape:
        raise RuntimeError(
            f"reconstruction shape mismatch: recon={tuple(recon.shape)} input={tuple(x.shape)}"
        )
    err = (recon - x).abs()

    save_video_view_grid(
        out_dir / "reconstruction" / "recon_grid.png",
        recon.detach().cpu(),
        max_frames=args.max_frames,
        max_views=args.max_views,
    )
    save_video_view_grid(
        out_dir / "reconstruction" / "abs_error.png",
        err.detach().cpu().clamp(0, 1),
        max_frames=args.max_frames,
        max_views=args.max_views,
    )

    metrics: dict[str, Any] = {
        "input": tensor_stats(x),
        "reconstruction_l1": float(F.l1_loss(recon, x).item()),
        "reconstruction_mse": float(F.mse_loss(recon, x).item()),
        "posterior_mean": tensor_stats(output["posterior"].mean),
        "posterior_logvar": tensor_stats(output["posterior"].logvar),
    }

    activation_stats = {}
    for name, tensor in activations.items():
        activation_stats[name] = save_activation_debug(
            out_dir, name, tensor, args.max_frames, args.max_views)
    metrics["activations"] = activation_stats

    metrics["projector"] = projector_debug(
        model, batch, args.use_camera_params, out_dir,
        args.max_frames, args.max_views, device)
    metrics["rope"] = rope_debug(out_dir, activations, model, args.max_views)

    write_json(out_dir / "debug_metrics.json", metrics)
    write_report(out_dir, args, metrics)

    print(f"[done] wrote visual debug report to {out_dir}")
    print(f"[done] open {out_dir / 'report.md'}")


if __name__ == "__main__":
    # Local Windows environments with both MKL and torch OpenMP sometimes need:
    # $env:KMP_DUPLICATE_LIB_OK='TRUE'
    main()
