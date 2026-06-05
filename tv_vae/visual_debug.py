"""Visual debug for the camera-conditioned TV VAE.

Outputs are grouped by the new architecture, not by the old cylinder path:

- input / reconstruction grids;
- camera ray and camera-center diagnostics;
- scene-token and latent-token statistics;
- activation summaries for encoder, scene aggregation, TV compression, and
  decoder stages.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from .data import (  # noqa: E402
        MultiViewVAEAdapter,
        SyntheticConfig,
        SyntheticCylinderDataset,
        make_nuscenes_base,
        make_nuscenes_scene_folder_base,
        vae_collate,
    )
    from .model import GeometryEmbedding, TVVAE  # noqa: E402
except ImportError:  # pragma: no cover
    from data import (  # noqa: E402
        MultiViewVAEAdapter,
        SyntheticConfig,
        SyntheticCylinderDataset,
        make_nuscenes_base,
        make_nuscenes_scene_folder_base,
        vae_collate,
    )
    from model import GeometryEmbedding, TVVAE  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "nuplan", "nus", "nuscenes", "nuscenes_scene"],
                   default="synthetic")
    p.add_argument("--out", default="tv_vae/runs/visual_debug")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--device", default="cuda")

    p.add_argument("--image-hw", type=int, nargs=2, default=[64, 128])
    p.add_argument("--sequence-length", type=int, default=5)
    p.add_argument("--view-count", type=int, default=6)

    p.add_argument("--nusc-data-root", default="")
    p.add_argument("--nusc-cache-root", default="")
    p.add_argument("--nusc-dataset-name", default="v1.0-trainval")
    p.add_argument("--nusc-split", default="train")
    p.add_argument("--nusc-fps", type=int, default=2)
    p.add_argument("--nusc-stride", type=float, default=0.5)
    p.add_argument("--nusc-keyframe-only", action="store_true", default=True)

    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--latent-channels", type=int, default=32)
    p.add_argument("--latent-time-count", type=int, default=2)
    p.add_argument("--scene-token-count", type=int, default=8)
    p.add_argument("--latent-scene-token-count", type=int, default=4)
    p.add_argument("--spatial-downsample-factor", type=int, default=4)
    p.add_argument("--num-attention-heads", type=int, default=4)
    p.add_argument("--tv-compression", choices=["joint", "non_joint"],
                   default="joint")
    p.add_argument("--max-views", type=int, default=6)
    p.add_argument("--max-frames", type=int, default=5)
    return p


def make_dataset(args):
    if args.data == "synthetic":
        return SyntheticCylinderDataset(SyntheticConfig(
            samples_per_epoch=max(args.sample_index + 1, 8),
            sequence_length=args.sequence_length,
            view_count=args.view_count,
            image_hw=tuple(args.image_hw),
        ))
    if args.data in ("nus", "nuscenes"):
        if not args.nusc_data_root:
            raise ValueError("--nusc-data-root is required for nuScenes.")
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
    if args.data == "nuscenes_scene":
        if not args.nusc_data_root:
            raise ValueError("--nusc-data-root is required for nuscenes_scene.")
        base = make_nuscenes_scene_folder_base(
            data_root=args.nusc_data_root,
            split=args.nusc_split,
            sequence_length=args.sequence_length,
        )
        return MultiViewVAEAdapter(
            base, sequence_length=args.sequence_length,
            image_hw=tuple(args.image_hw))

    from dwm.tools.dataset_nus import make_base_ds
    base = make_base_ds(train=True)
    return MultiViewVAEAdapter(
        base, sequence_length=args.sequence_length,
        image_hw=tuple(args.image_hw))


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    x = x.detach().float()
    finite = torch.isfinite(x)
    if finite.any():
        y = x[finite]
        return {
            "shape": list(x.shape),
            "finite_fraction": float(finite.float().mean().item()),
            "min": float(y.min().item()),
            "max": float(y.max().item()),
            "mean": float(y.mean().item()),
            "std": float(y.std(unbiased=False).item()),
            "zero_fraction": float((x == 0).float().mean().item()),
        }
    return {
        "shape": list(x.shape),
        "finite_fraction": 0.0,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "zero_fraction": None,
    }


def save_image_grid(path: Path, x: torch.Tensor, nrow: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu().float().clamp(-1, 1)
    x = (x + 1) / 2
    vutils.save_image(x, str(path), nrow=nrow)


def save_heatmap(path: Path, x: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu().float()
    if x.ndim == 2:
        x = x[None]
    if x.ndim == 3 and x.shape[0] != 1:
        x = x.mean(dim=0, keepdim=True)
    x_min = x.amin()
    x_max = x.amax()
    x = (x - x_min) / (x_max - x_min).clamp(min=1e-6)
    vutils.save_image(x, str(path))


def save_matrix(path: Path, x: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu().float()
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    x_min = x.amin()
    x_max = x.amax()
    x = (x - x_min) / (x_max - x_min).clamp(min=1e-6)
    x = x[None, None]
    x = F.interpolate(x, size=(256, 256), mode="nearest")
    vutils.save_image(x, str(path))


def load_checkpoint_if_needed(model: TVVAE, path: str):
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[checkpoint] loaded {path}")
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}")


def capture_activations(model: TVVAE):
    activations: dict[str, torch.Tensor] = {}
    handles = []
    modules = {
        "stem": model.stem,
        "geometry": model.geometry,
        "scene_aggregator": model.scene_aggregator,
        "compress": model.compress,
        "to_moments": model.to_moments,
        "from_latent": model.from_latent,
        "expand": model.expand,
        "camera_renderer": model.camera_renderer,
        "head": model.head,
    }
    for i, block in enumerate(model.down_blocks):
        modules[f"down_{i}"] = block
    for i, block in enumerate(model.up_blocks):
        modules[f"up_{i}"] = block

    def make_hook(name):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                activations[name] = output.detach().cpu()
        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    return activations, handles


def geometry_debug(
    out_dir: Path,
    x: torch.Tensor,
    model: TVVAE,
    K: torch.Tensor | None,
    E: torch.Tensor | None,
    intr_hw,
) -> dict[str, Any]:
    b, t, v, _c, h, w = x.shape
    h_lat = max(1, h // int(model.config.spatial_downsample_factor))
    w_lat = max(1, w // int(model.config.spatial_downsample_factor))
    raw = GeometryEmbedding._camera_features(
        b, t, v, h_lat, w_lat, torch.float32, x.device, K, E, intr_hw)
    uv = raw[..., :2]
    ray = raw[..., 2:5]
    center = raw[..., 5:8]
    stats = {
        "has_camera_params": bool(K is not None and E is not None),
        "raw_geometry": tensor_stats(raw),
        "uv": tensor_stats(uv),
        "ray_ego": tensor_stats(ray),
        "camera_center_ego": tensor_stats(center),
    }
    if K is not None:
        stats["intrinsics"] = {
            "fx_mean": float(K[..., 0, 0].float().mean().item()),
            "fy_mean": float(K[..., 1, 1].float().mean().item()),
            "cx_mean": float(K[..., 0, 2].float().mean().item()),
            "cy_mean": float(K[..., 1, 2].float().mean().item()),
            "intrinsics_hw": list(intr_hw) if intr_hw is not None else None,
        }
    if E is not None:
        centers = E[..., :3, 3].float()
        stats["extrinsics"] = {
            "center_mean": [float(v) for v in centers.mean(dim=(0, 1, 2)).tolist()],
            "center_std": [float(v) for v in centers.std(dim=(0, 1, 2), unbiased=False).tolist()],
        }

    view_keep = min(v, 6)
    ray_rgb = ray[0, 0, :view_keep].permute(0, 3, 1, 2).detach().cpu()
    save_image_grid(out_dir / "geometry" / "ray_ego_rgb.png", ray_rgb, view_keep)
    center_norm = center[0, 0, :view_keep].norm(dim=-1).detach().cpu()
    save_image_grid(
        out_dir / "geometry" / "camera_center_norm.png",
        center_norm[:, None].expand(-1, 3, -1, -1) * 2 - 1,
        view_keep)
    return stats


def scene_latent_debug(out_dir: Path, activations: dict[str, torch.Tensor],
                       posterior, model: TVVAE) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for name in ("scene_aggregator", "compress", "to_moments",
                 "from_latent", "expand", "camera_renderer"):
        if name in activations:
            stats[name] = tensor_stats(activations[name])

    mean = posterior.mean.detach().float().cpu()
    logvar = posterior.logvar.detach().float().cpu()
    stats["posterior_mean"] = tensor_stats(mean)
    stats["posterior_logvar"] = tensor_stats(logvar)
    stats["latent_budget"] = {
        "tv_compression": model.config.tv_compression,
        "scene_token_count": int(model.config.scene_token_count),
        "latent_time_count": int(model.config.latent_time_count),
        "latent_scene_token_count": int(model.config.latent_scene_token_count),
        "latent_spatial_shape": list(mean.shape[-2:]),
        "latent_token_count": int(mean.shape[1] * mean.shape[2] *
                                  mean.shape[-2] * mean.shape[-1]),
    }

    # Token energy maps help spot dead scene tokens or collapsed latent slots.
    if "scene_aggregator" in activations:
        scene = activations["scene_aggregator"].float()
        # [B,T,S,C,H,W] -> [T,S]
        energy = scene.pow(2).mean(dim=(0, 3, 4, 5))
        save_matrix(out_dir / "tokens" / "scene_token_energy_TxS.png", energy)
        stats["scene_token_energy_TxS"] = tensor_stats(energy)
    latent_energy = mean.pow(2).mean(dim=(0, 3, 4, 5))
    save_matrix(out_dir / "tokens" / "latent_token_energy_TxS.png", latent_energy)
    stats["latent_token_energy_TxS"] = tensor_stats(latent_energy)

    # Spatial energy detects whether the bottleneck is using only a few cells.
    spatial_energy = mean.pow(2).mean(dim=(0, 1, 2, 3))
    save_heatmap(out_dir / "tokens" / "latent_spatial_energy.png", spatial_energy)
    return stats


def activation_debug(out_dir: Path, activations: dict[str, torch.Tensor]) -> dict[str, Any]:
    stats = {}
    for name, tensor in activations.items():
        stats[name] = tensor_stats(tensor)
        if tensor.ndim >= 4:
            x = tensor.float()
            spatial = x.pow(2).mean(dim=tuple(range(x.ndim - 2)))
            save_heatmap(out_dir / "activations" / f"{name}_spatial_energy.png", spatial)
    return stats


def write_report(out_dir: Path, metrics: dict[str, Any]):
    lines = [
        "# TV VAE Visual Debug Report",
        "",
        "## Key Files",
        "- `input/input_grid.png`: raw target frames.",
        "- `reconstruction/recon_grid.png`: VAE reconstruction.",
        "- `reconstruction/abs_error.png`: absolute camera-domain error.",
        "- `geometry/ray_ego_rgb.png`: ego-frame ray directions encoded as RGB.",
        "- `geometry/camera_center_norm.png`: camera-center magnitude map.",
        "- `tokens/scene_token_energy_TxS.png`: scene-token usage before TV compression.",
        "- `tokens/latent_token_energy_TxS.png`: latent token usage after TV compression.",
        "- `tokens/latent_spatial_energy.png`: spatial latent energy map.",
        "- `activations/*_spatial_energy.png`: stage-wise activation energy.",
        "",
        "## How To Read",
        "- Bad input grid means dataset/adapter trouble.",
        "- Missing or constant ray maps mean K/E are absent or malformed.",
        "- Dead scene-token energy rows/columns mean scene aggregation collapsed.",
        "- Dead latent token rows/columns mean TV compression collapsed.",
        "- Sharp input but blurry recon with healthy tokens usually means the bottleneck is too strong or decoder capacity is low.",
        "- High error concentrated in one camera means camera order or camera parameters are suspicious.",
        "",
        "## Metrics Summary",
        "```json",
        json.dumps(metrics, indent=2, ensure_ascii=True)[:12000],
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(args)
    sample = dataset[args.sample_index]
    batch = vae_collate([sample])
    x = batch["vae_images"].to(device)
    K = batch.get("camera_intrinsics")
    E = batch.get("camera_transforms")
    intr_hw = batch.get("intrinsics_hw")
    if K is not None and E is not None:
        K = K.to(device)
        E = E.to(device)
    else:
        K = None
        E = None

    v_input = int(x.shape[2])
    model = TVVAE(
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        sequence_length=args.sequence_length,
        view_count=v_input,
        scene_token_count=args.scene_token_count,
        latent_time_count=args.latent_time_count,
        latent_scene_token_count=args.latent_scene_token_count,
        spatial_downsample_factor=args.spatial_downsample_factor,
        num_attention_heads=args.num_attention_heads,
        tv_compression=args.tv_compression,
        image_height=args.image_hw[0],
        image_width=args.image_hw[1],
    ).to(device)
    load_checkpoint_if_needed(model, args.checkpoint)
    model.eval()

    activations, handles = capture_activations(model)
    with torch.no_grad():
        output = model(
            x,
            sample_posterior=False,
            intrinsics=K,
            extrinsics=E,
            intrinsics_hw=intr_hw,
        )
    for handle in handles:
        handle.remove()

    recon = output["sample"]
    posterior = output["posterior"]
    metrics: dict[str, Any] = {
        "run_args": vars(args),
        "input": tensor_stats(x),
        "reconstruction": tensor_stats(recon),
        "reconstruction_l1": float((recon - x).abs().mean().item()),
        "reconstruction_mse": float(F.mse_loss(recon, x).item()),
    }

    max_t = min(args.max_frames, x.shape[1])
    max_v = min(args.max_views, x.shape[2])
    input_grid = x[0, :max_t, :max_v].reshape(max_t * max_v, *x.shape[-3:])
    recon_grid = recon[0, :max_t, :max_v].reshape(max_t * max_v, *recon.shape[-3:])
    err = (recon - x).abs().mean(dim=3, keepdim=True).expand(-1, -1, -1, 3, -1, -1)
    err_grid = err[0, :max_t, :max_v].reshape(max_t * max_v, 3, x.shape[-2], x.shape[-1])
    save_image_grid(out_dir / "input" / "input_grid.png", input_grid, max_v)
    save_image_grid(out_dir / "reconstruction" / "recon_grid.png", recon_grid, max_v)
    save_image_grid(out_dir / "reconstruction" / "abs_error.png", err_grid * 2 - 1, max_v)

    if x.shape[1] > 1:
        delta = (x[:, 1:] - x[:, :-1]).abs().mean(dim=3, keepdim=True)
        delta = delta.expand(-1, -1, -1, 3, -1, -1)
        delta_grid = delta[0, :max_t - 1, :max_v].reshape(
            max(max_t - 1, 0) * max_v, 3, x.shape[-2], x.shape[-1])
        if delta_grid.numel() > 0:
            save_image_grid(
                out_dir / "input" / "temporal_delta.png",
                delta_grid * 2 - 1,
                max_v)

    metrics["geometry"] = geometry_debug(out_dir, x, model, K, E, intr_hw)
    metrics["tokens"] = scene_latent_debug(out_dir, activations, posterior, model)
    metrics["activations"] = activation_debug(out_dir, activations)

    (out_dir / "debug_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True),
        encoding="utf-8")
    write_report(out_dir, metrics)
    print(f"[done] visual debug written to {out_dir}")


if __name__ == "__main__":
    main()
