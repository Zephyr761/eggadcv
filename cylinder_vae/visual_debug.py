"""Visual debugging for CrossView4DVAE.

This script runs one batch through the VAE and writes a compact debug report:

- input / reconstruction / absolute error grids;
- temporal input deltas;
- activation statistics and per-stage energy maps;
- per-stage cross-view cosine similarity matrices;
- layered projector source-camera coverage maps using synthetic one-hot
  camera IDs;
- projector-only camera->two-layer-cylinder->camera roundtrip diagnostics;
- near/far latent budget and T-V joint compression energy maps;
- seam heatmaps for target and decoded cylinder layers.

Everything is written under ``step_1/runs`` by default.
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

try:
    from .crossview_vae import (  # noqa: E402
        CrossView4DVAE,
        _build_view_coords,
    )
    from .data import (  # noqa: E402
        MultiViewVAEAdapter,
        SyntheticConfig,
        SyntheticCylinderDataset,
        make_nuscenes_base,
        make_nuscenes_scene_folder_base,
        vae_collate,
    )
except ImportError:
    from crossview_vae import (  # noqa: E402
        CrossView4DVAE,
        _build_view_coords,
    )
    from data import (  # noqa: E402
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

VIEW_ACTIVATION_STAGES = {
    "virtual_projector",
    "down1",
    "view_mix1",
    "down2",
    "view_mix2",
    "down3",
    "view_mix3",
    "pre_attn",
    "view_up",
    "post_attn",
    "up1",
    "view_mix_up1",
    "up2",
    "view_mix_up2",
    "up3",
}

PACKED_LATENT_STAGES = {
    "view_down",
    "attn",
    "to_moments",
    "from_latent",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", choices=["synthetic", "nus", "nuscenes", "nuscenes_scene", "nuplan"],
                   default="synthetic")
    p.add_argument("--out", default="cylinder_vae/runs/visual_debug")
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
    p.add_argument("--latent-time-count", type=int, default=2)
    p.add_argument("--near-latent-time-count", type=int, default=-1)
    p.add_argument("--far-latent-time-count", type=int, default=-1)
    p.add_argument("--near-latent-view-count", type=int, default=-1)
    p.add_argument("--far-latent-view-count", type=int, default=-1)
    p.add_argument("--latent-spatial-downsample-factor", type=int, default=1)
    p.add_argument("--near-latent-spatial-downsample-factor", type=int, default=-1)
    p.add_argument("--far-latent-spatial-downsample-factor", type=int, default=-1)
    p.add_argument("--num-attention-heads", type=int, default=2)
    p.add_argument("--num-bottleneck-blocks", type=int, default=1)
    p.add_argument("--temporal-downsample-factor", type=int, default=4)
    p.add_argument("--temporal-pre", type=int, default=1)
    p.add_argument("--spatial-downsample-factor", type=int, default=8)
    p.add_argument("--cylinder-radii", type=float, nargs="+", default=[4.0, 20.0])
    p.add_argument("--cylinder-height-scale", type=float, default=1.0,
                   help="Vertical range scale for the canonical cylinder. "
                        "Match the value used for training; values above 1 "
                        "expand camera render coverage.")
    p.add_argument("--cylinder-vertical-mode",
                   choices=["aspect", "camera_fov"], default="camera_fov",
                   help="Match the training vertical range mode. camera_fov "
                        "uses intrinsics to reduce unobserved gray bands.")
    p.add_argument("--projector-edge-feather", type=float, default=0.05,
                   help="Soft fade width near source-image borders. Match "
                        "the value used for training.")
    p.add_argument("--projector-angular-power", type=float, default=0.5,
                   help="Power applied to angular visibility weights. Values "
                        "below 1 soften source-camera transitions.")
    p.add_argument("--projector-blend-mode",
                   choices=["soft", "best_camera"], default="soft",
                   help="Match the training projector blend mode. "
                        "best_camera avoids RGB ghosting in overlap regions.")
    p.add_argument("--ego-coordinate-mode", choices=["nuscenes", "synthetic"],
                   default="nuscenes")
    p.add_argument("--decode-camera-from-cylinder", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--use-view-residual", action="store_true",
                   help=argparse.SUPPRESS)

    p.add_argument("--use-camera-params", action="store_true",
                   help="Use K/E for closed-loop camera->cylinder->latent->"
                        "cylinder->camera reconstruction and projector diagnostics.")
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
    return heatmap_rgb01(y)


def heatmap_rgb01(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().float().clamp(0, 1)
    r = y
    g = (1.0 - (2.0 * y - 1.0).abs()).clamp(0, 1)
    b = 1.0 - y
    return torch.stack([r, g, b], dim=0)


def matrix_heatmap(
    mat: torch.Tensor,
    cell: int = 28,
    max_side: int = 96,
    max_pixels: int = 2048,
) -> torch.Tensor:
    if mat.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={tuple(mat.shape)}")
    h, w = int(mat.shape[-2]), int(mat.shape[-1])
    if max(h, w) > max_side:
        mat = F.interpolate(
            mat[None, None].float(),
            size=(min(h, max_side), min(w, max_side)),
            mode="area",
        )[0, 0]
        h, w = int(mat.shape[-2]), int(mat.shape[-1])
    cell = max(1, min(int(cell), max_pixels // max(max(h, w), 1)))
    m = heatmap_rgb(mat)
    return F.interpolate(
        m.unsqueeze(0), scale_factor=cell, mode="nearest").squeeze(0)


def save_image_grid(path: Path, images: torch.Tensor, nrow: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images.detach().cpu(), str(path), nrow=nrow)


def save_video_view_grid(path: Path, x: torch.Tensor, max_frames: int,
                         max_views: int, value_range: str = "auto") -> None:
    # x: [B,T,V,3,H,W] or [B,T,D,V,3,H,W] in [-1,1] or [0,1].
    y = x.detach().cpu()[0]
    if y.ndim == 6:
        t0, d, v0, c, h, w = y.shape
        y = y.reshape(t0, d * v0, c, h, w)
    t = min(y.shape[0], max_frames)
    v = min(y.shape[1], max_views)
    y = y[:t, :v]
    if value_range == "pm1" or (value_range == "auto" and y.min() < -0.05):
        y = (y.clamp(-1, 1) + 1.0) / 2.0
    elif value_range not in {"auto", "01"}:
        raise ValueError(f"Unknown value_range={value_range!r}")
    y = y.clamp(0, 1).reshape(t * v, y.shape[2], y.shape[3], y.shape[4])
    save_image_grid(path, y, nrow=v)


def flatten_layer_view(x: torch.Tensor) -> torch.Tensor:
    """Flatten a layered cylinder from D,V to a single view-like axis."""
    if x.ndim == 7:
        b, t, d, v, c, h, w = x.shape
        return x.reshape(b, t, d * v, c, h, w)
    if x.ndim == 6:
        return x
    raise ValueError(f"Expected 6D or 7D tensor, got shape={tuple(x.shape)}")


def layer_labels(model: CrossView4DVAE) -> list[str]:
    radii = list(getattr(model.virtual_projector, "cylinder_radii", []))
    labels = []
    for i, radius in enumerate(radii):
        name = "near" if i == 0 else ("far" if i == 1 else f"layer{i}")
        labels.append(f"{i:02d}_{name}_r{float(radius):g}")
    return labels or ["00_layer"]


def save_layered_video_grids(path: Path, x: torch.Tensor,
                             model: CrossView4DVAE, max_frames: int,
                             max_views: int,
                             value_range: str = "auto") -> None:
    """Save flattened D*V overview and per-layer grids."""
    save_video_view_grid(path, x, max_frames, max_views, value_range)
    if x.ndim != 7:
        return
    for layer_idx, label in enumerate(layer_labels(model)[:x.shape[2]]):
        per_layer = x[:, :, layer_idx]
        layer_path = path.with_name(f"{path.stem}_{label}{path.suffix}")
        save_video_view_grid(
            layer_path, per_layer, max_frames, max_views, value_range)


def save_tensor_heat_grid(path: Path, maps: torch.Tensor, max_frames: int,
                          max_views: int, fixed01: bool = False) -> None:
    # maps: [T,V,H,W]
    t = min(maps.shape[0], max_frames)
    v = min(maps.shape[1], max_views)
    mapper = heatmap_rgb01 if fixed01 else heatmap_rgb
    imgs = [mapper(maps[i, j]) for i in range(t) for j in range(v)]
    save_image_grid(path, torch.stack(imgs), nrow=v)


def save_layered_heat_grids(path: Path, maps: torch.Tensor,
                            model: CrossView4DVAE, max_frames: int,
                            max_views: int, fixed01: bool = False) -> None:
    # maps: [T,V,H,W] or [T,D,V,H,W]
    if maps.ndim == 5:
        t, d, v, h, w = maps.shape
        flat = maps.reshape(t, d * v, h, w)
        save_tensor_heat_grid(path, flat, max_frames, max_views, fixed01=fixed01)
        for layer_idx, label in enumerate(layer_labels(model)[:d]):
            layer_path = path.with_name(f"{path.stem}_{label}{path.suffix}")
            save_tensor_heat_grid(
                layer_path, maps[:, layer_idx], max_frames, max_views,
                fixed01=fixed01)
        return
    save_tensor_heat_grid(path, maps, max_frames, max_views, fixed01=fixed01)


def as_6d_activation(name: str, x: torch.Tensor) -> torch.Tensor | None:
    if name not in VIEW_ACTIVATION_STAGES:
        return None
    if x.ndim == 7:
        return flatten_layer_view(x)
    if x.ndim != 6:
        return None
    return x


def as_packed_latent_activation(name: str, x: torch.Tensor) -> torch.Tensor | None:
    if name not in PACKED_LATENT_STAGES or x.ndim != 6:
        return None
    # view_down / attn: [B,1,N,C,1,1]
    if name in {"view_down", "attn"} and x.shape[1] == 1 and x.shape[-2:] == (1, 1):
        return x[:, 0, :, :, 0, 0].contiguous()
    # Linear hooks keep channels last:
    # to_moments / from_latent: [B,1,N,1,1,C]
    if name in {"to_moments", "from_latent"} and x.shape[1] == 1 and x.shape[3:5] == (1, 1):
        return x[:, 0, :, 0, 0, :].contiguous()
    return None


def save_vector_heatmap(path: Path, values: torch.Tensor, max_width: int = 512) -> None:
    y = values.detach().float().reshape(1, 1, 1, -1)
    if y.shape[-1] > max_width:
        y = F.interpolate(y, size=(1, max_width), mode="area")
    img = heatmap_rgb(y[0, 0, 0]).unsqueeze(1)
    save_image_grid(path, img.unsqueeze(0), nrow=1)


def view_cosine_matrix(
    act: torch.Tensor,
    max_items: int = 64,
) -> tuple[torch.Tensor, dict[str, Any]]:
    # act: [B,T,V,C,H,W]. Average batch/time/spatial, compare view feature vecs.
    y = act.detach().float()
    full_items = int(y.shape[2])
    sampled = False
    if full_items > max_items:
        idx = torch.linspace(0, full_items - 1, max_items).long()
        y = y.index_select(2, idx.to(y.device))
        sampled = True
    y = y.mean(dim=(0, 1, 4, 5))  # [V,C]
    y = F.normalize(y, dim=-1, eps=1e-6)
    info = {
        "full_items": full_items,
        "items_used": int(y.shape[0]),
        "sampled": sampled,
    }
    return y @ y.t(), info


def save_activation_debug(out_dir: Path, name: str, tensor: torch.Tensor,
                          max_frames: int, max_views: int) -> dict[str, Any]:
    stats = tensor_stats(tensor)
    packed = as_packed_latent_activation(name, tensor)
    if packed is not None:
        stats["activation_semantic"] = "packed_latent_tokens"
        stats["token_count"] = int(packed.shape[1])
        stats["channel_count"] = int(packed.shape[2])
        token_energy = packed.detach().float().abs().mean(dim=(0, 2))
        stats["token_energy"] = {
            "min": float(token_energy.min().item()),
            "max": float(token_energy.max().item()),
            "mean": float(token_energy.mean().item()),
            "std": float(token_energy.std(unbiased=False).item()),
        }
        save_vector_heatmap(
            out_dir / "activations" / f"{name}_token_energy.png",
            token_energy,
        )
        return stats

    act = as_6d_activation(name, tensor)
    if act is None:
        return stats

    stats["activation_semantic"] = "view_like"
    energy = act.detach().float().abs().mean(dim=3)[0]  # [T,V,H,W]
    stats["per_time_view_energy_shape"] = list(energy.shape[:2])
    stats["per_time_view_energy"] = (
        energy.mean(dim=(-2, -1))[:max_frames, :max_views]
        .detach().cpu().tolist()
    )
    save_tensor_heat_grid(
        out_dir / "activations" / f"{name}_energy.png",
        energy,
        max_frames=max_frames,
        max_views=max_views,
    )
    cos, cos_info = view_cosine_matrix(act, max_items=max(max_views, 64))
    stats["view_cosine_info"] = cos_info
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


def camera_geometry_debug(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    E = batch.get("camera_transforms")
    K = batch.get("camera_intrinsics")
    if E is None or K is None:
        return {"available": False}
    E = E.to(device).detach().float()
    K = K.to(device).detach().float()
    # First batch / first frame is enough to expose convention and ordering bugs.
    E0 = E[0, 0]
    K0 = K[0, 0]
    cam_pos = E0[:, :3, 3]
    cam_forward = F.normalize(E0[:, :3, 2], dim=-1, eps=1e-6)
    cam_up = F.normalize(-E0[:, :3, 1], dim=-1, eps=1e-6)
    az_xy = torch.rad2deg(torch.atan2(cam_forward[:, 1], cam_forward[:, 0]))
    az_xz = torch.rad2deg(torch.atan2(cam_forward[:, 0], cam_forward[:, 2]))
    return {
        "available": True,
        "note": (
            "azimuth_xy_deg assumes nuScenes ego convention: x/y ground plane, "
            "z up. azimuth_xz_deg matches the old synthetic projector convention: "
            "x/z ground plane, y up. Large disagreement means the projector "
            "coordinate convention matters."
        ),
        "position_ego_first_frame": cam_pos.cpu().tolist(),
        "forward_ego_first_frame": cam_forward.cpu().tolist(),
        "up_ego_first_frame": cam_up.cpu().tolist(),
        "azimuth_xy_deg_first_frame": az_xy.cpu().tolist(),
        "azimuth_xz_deg_first_frame": az_xz.cpu().tolist(),
        "intrinsics_first_frame": K0.cpu().tolist(),
    }


def _projector_source_stats(projected: torch.Tensor, out_dir: Path, prefix: str,
                            max_frames: int, max_views: int,
                            model: CrossView4DVAE | None = None) -> dict[str, Any]:
    # projected: [B,T,V_virtual,C_source,H,W].
    if projected.ndim == 7:
        projected_flat = flatten_layer_view(projected)
    else:
        projected_flat = projected
    weights = projected_flat[0].detach().float().clamp(min=0)
    v_source = weights.shape[2]
    coverage = weights.sum(dim=2)  # [T,V_virtual,H,W]
    probs = weights / coverage.unsqueeze(2).clamp(min=1e-8)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=2)
    if v_source > 1:
        entropy = entropy / math.log(v_source)

    save_tensor_heat_grid(
        out_dir / "projector" / f"{prefix}_coverage.png",
        coverage,
        max_frames=max_frames,
        max_views=max_views,
        fixed01=True,
    )
    if prefix == "active":
        save_tensor_heat_grid(
            out_dir / "projector" / "coverage.png",
            coverage,
            max_frames=max_frames,
            max_views=max_views,
            fixed01=True,
        )
    save_tensor_heat_grid(
        out_dir / "projector" / f"{prefix}_source_entropy.png",
        entropy,
        max_frames=max_frames,
        max_views=max_views,
    )
    if prefix == "active":
        save_tensor_heat_grid(
            out_dir / "projector" / "source_entropy.png",
            entropy,
            max_frames=max_frames,
            max_views=max_views,
        )

    palette = camera_palette(v_source).to(weights.device)
    rgb = torch.einsum("tvchw,cr->tvrhw", probs, palette)
    t_keep = min(rgb.shape[0], max_frames)
    v_keep = min(rgb.shape[1], max_views)
    rgb_imgs = rgb[:t_keep, :v_keep].reshape(
        t_keep * v_keep, 3, rgb.shape[-2], rgb.shape[-1])
    save_image_grid(out_dir / "projector" / f"{prefix}_source_camera_rgb.png",
                    rgb_imgs, nrow=v_keep)
    if prefix == "active":
        save_image_grid(out_dir / "projector" / "source_camera_rgb.png",
                        rgb_imgs, nrow=v_keep)

    per_layer = {}
    if projected.ndim == 7 and model is not None:
        labels = layer_labels(model)
        d = projected.shape[2]
        for layer_idx in range(d):
            layer = projected[:, :, layer_idx]
            layer_weights = layer[0].detach().float().clamp(min=0)
            layer_coverage = layer_weights.sum(dim=2)
            layer_probs = layer_weights / layer_coverage.unsqueeze(2).clamp(min=1e-8)
            layer_entropy = -(
                layer_probs * layer_probs.clamp(min=1e-8).log()).sum(dim=2)
            if v_source > 1:
                layer_entropy = layer_entropy / math.log(v_source)
            label = labels[layer_idx] if layer_idx < len(labels) else f"{layer_idx:02d}"
            save_tensor_heat_grid(
                out_dir / "projector" / f"{prefix}_coverage_{label}.png",
                layer_coverage,
                max_frames=max_frames,
                max_views=max_views,
                fixed01=True,
            )
            save_tensor_heat_grid(
                out_dir / "projector" / f"{prefix}_source_entropy_{label}.png",
                layer_entropy,
                max_frames=max_frames,
                max_views=max_views,
            )
            layer_rgb = torch.einsum("tvchw,cr->tvrhw", layer_probs, palette)
            lt = min(layer_rgb.shape[0], max_frames)
            lv = min(layer_rgb.shape[1], max_views)
            layer_imgs = layer_rgb[:lt, :lv].reshape(
                lt * lv, 3, layer_rgb.shape[-2], layer_rgb.shape[-1])
            save_image_grid(
                out_dir / "projector" / f"{prefix}_source_camera_rgb_{label}.png",
                layer_imgs,
                nrow=lv,
            )
            per_layer[label] = {
                "coverage_min": float(layer_coverage.min().item()),
                "coverage_mean": float(layer_coverage.mean().item()),
                "coverage_zero_fraction": float(
                    (layer_coverage < 1e-4).float().mean().item()),
                "entropy_mean": float(layer_entropy.mean().item()),
                "entropy_p95": float(
                    torch.quantile(layer_entropy.flatten(), 0.95).item()),
            }

    dominant = probs.mean(dim=(-2, -1)).argmax(dim=-1)  # [T,V_virtual]
    return {
        "weights_shape": list(projected.shape),
        "coverage_min": float(coverage.min().item()),
        "coverage_mean": float(coverage.mean().item()),
        "coverage_zero_fraction": float((coverage < 1e-4).float().mean().item()),
        "entropy_mean": float(entropy.mean().item()),
        "entropy_p95": float(torch.quantile(entropy.flatten(), 0.95).item()),
        "dominant_source_by_time_virtual_view": dominant.cpu().tolist(),
        "per_layer": per_layer,
    }


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
        fallback_ids = model.virtual_projector.project_layers(
            ids, None, None, intr_hw)
        fallback_rgb = model.virtual_projector.project_layers(
            x, None, None, intr_hw)
        camera_ids = None
        camera_rgb = None
        camera_mask = None
        if K is not None and E is not None:
            camera_ids = model.virtual_projector.project_layers(
                ids, K, E, intr_hw)
            camera_rgb = model.virtual_projector.project_layers(
                x, K, E, intr_hw)
            ones = torch.ones(b, t, v, 1, h, w, device=device, dtype=x.dtype)
            camera_mask = model.virtual_projector.project_layers(
                ones, K, E, intr_hw)

    fallback_stats = _projector_source_stats(
        fallback_ids, out_dir, "fallback", max_frames, max_views, model)
    save_layered_video_grids(
        out_dir / "projector" / "fallback_projected_rgb.png",
        fallback_rgb.detach().cpu(),
        model,
        max_frames=max_frames,
        max_views=max_views,
        value_range="pm1",
    )
    # Back-compatible filenames: point to whichever path the model used.
    active_ids = camera_ids if camera_ids is not None else fallback_ids
    active_rgb = camera_rgb if camera_rgb is not None else fallback_rgb
    active_stats = _projector_source_stats(
        active_ids, out_dir, "active", max_frames, max_views, model)
    save_layered_video_grids(
        out_dir / "projector" / "active_projected_rgb.png",
        active_rgb.detach().cpu(),
        model,
        max_frames=max_frames,
        max_views=max_views,
        value_range="pm1",
    )

    stats = {
        "projector_used_camera_params": bool(K is not None and E is not None),
        "fallback": fallback_stats,
        "active": active_stats,
        "camera_geometry": camera_geometry_debug(batch, device),
    }
    if camera_ids is not None and camera_rgb is not None:
        camera_stats = _projector_source_stats(
            camera_ids, out_dir, "camera", max_frames, max_views, model)
        save_layered_video_grids(
            out_dir / "projector" / "camera_projected_rgb.png",
            camera_rgb.detach().cpu(),
            model,
            max_frames=max_frames,
            max_views=max_views,
            value_range="pm1",
        )
        diff = (camera_rgb.detach().float() - fallback_rgb.detach().float()).abs()
        save_layered_video_grids(
            out_dir / "projector" / "camera_vs_fallback_absdiff.png",
            diff.cpu().clamp(0, 1),
            model,
            max_frames=max_frames,
            max_views=max_views,
            value_range="01",
        )
        stats["camera"] = camera_stats
        stats["camera_vs_fallback_l1"] = float(diff.mean().item())
        if camera_mask is not None:
            mask_maps = camera_mask[0, :, :, :, 0].detach().float().cpu()
            save_layered_heat_grids(
                out_dir / "projector" / "camera_projected_mask.png",
                mask_maps,
                model,
                max_frames=max_frames,
                max_views=max_views,
                fixed01=True,
            )
            stats["camera_projected_mask"] = {
                "shape": list(camera_mask.shape),
                "mean": float(camera_mask.mean().item()),
                "zero_fraction": float(
                    (camera_mask < 1e-4).float().mean().item()),
            }
    return stats


def as_layered_cylinder(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 6:
        return x.unsqueeze(2)
    if x.ndim == 7:
        return x
    raise ValueError(f"Expected cylinder tensor, got shape={tuple(x.shape)}")


def per_layer_l1(pred: torch.Tensor, target: torch.Tensor,
                 model: CrossView4DVAE) -> dict[str, float]:
    p = as_layered_cylinder(pred).detach().float()
    t = as_layered_cylinder(target).detach().float()
    labels = layer_labels(model)
    vals = (p - t).abs().mean(dim=(0, 1, 3, 4, 5, 6))
    return {
        labels[i] if i < len(labels) else f"{i:02d}": float(vals[i].item())
        for i in range(vals.numel())
    }


def cylinder_seam_maps(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    # Returns [T,D,V,H,8] heat maps for right-edge to next-left-edge seams.
    y = as_layered_cylinder(x).detach().float()
    right = y[..., -1]  # [B,T,D,V,C,H]
    left_next = torch.roll(y, shifts=-1, dims=3)[..., 0]
    seam = (right - left_next).abs().mean(dim=4)  # [B,T,D,V,H]
    maps = seam[0].unsqueeze(-1).expand(-1, -1, -1, -1, 8).cpu()
    per_layer = seam.mean(dim=(0, 1, 3, 4))
    per_layer_p95 = torch.quantile(
        seam.permute(2, 0, 1, 3, 4).reshape(seam.shape[2], -1),
        0.95,
        dim=1,
    )
    return maps, {
        "mean": float(seam.mean().item()),
        "p95": float(torch.quantile(seam.flatten(), 0.95).item()),
        "per_layer_mean": [float(v.item()) for v in per_layer],
        "per_layer_p95": [float(v.item()) for v in per_layer_p95],
    }


def cylinder_debug(model: CrossView4DVAE, x: torch.Tensor,
                   output: dict[str, Any], K: torch.Tensor | None,
                   E: torch.Tensor | None, intr_hw: Any, out_dir: Path,
                   max_frames: int, max_views: int) -> dict[str, Any]:
    pred = output["cylinder_sample"]
    with torch.no_grad():
        target = model.virtual_projector.project_layers(x, K, E, intr_hw)
        mask = model.virtual_projector.project_layers(
            torch.ones_like(x[:, :, :, :1]), K, E, intr_hw)

    save_layered_video_grids(
        out_dir / "cylinder" / "target_cylinder_grid.png",
        target.detach().cpu(),
        model,
        max_frames=max_frames,
        max_views=max_views,
        value_range="pm1",
    )
    save_layered_video_grids(
        out_dir / "cylinder" / "pred_cylinder_grid.png",
        pred.detach().cpu(),
        model,
        max_frames=max_frames,
        max_views=max_views,
        value_range="pm1",
    )
    cyl_err = (pred.detach().float() - target.detach().float()).abs()
    save_layered_video_grids(
        out_dir / "cylinder" / "abs_error.png",
        cyl_err.cpu().clamp(0, 1),
        model,
        max_frames=max_frames,
        max_views=max_views,
        value_range="01",
    )
    mask_maps = mask[0, :, :, :, 0].detach().float().cpu()
    save_layered_heat_grids(
        out_dir / "cylinder" / "valid_mask.png",
        mask_maps,
        model,
        max_frames=max_frames,
        max_views=max_views,
        fixed01=True,
    )

    target_seams, target_seam_stats = cylinder_seam_maps(target)
    pred_seams, pred_seam_stats = cylinder_seam_maps(pred)
    err_seams, err_seam_stats = cylinder_seam_maps(cyl_err)
    save_layered_heat_grids(
        out_dir / "cylinder" / "target_seams.png",
        target_seams,
        model,
        max_frames=max_frames,
        max_views=max_views,
    )
    save_layered_heat_grids(
        out_dir / "cylinder" / "pred_seams.png",
        pred_seams,
        model,
        max_frames=max_frames,
        max_views=max_views,
    )
    save_layered_heat_grids(
        out_dir / "cylinder" / "error_seams.png",
        err_seams,
        model,
        max_frames=max_frames,
        max_views=max_views,
    )

    stats: dict[str, Any] = {
        "target": tensor_stats(target),
        "prediction": tensor_stats(pred),
        "mask": tensor_stats(mask),
        "l1": float(cyl_err.mean().item()),
        "per_layer_l1": per_layer_l1(pred, target, model),
        "target_seams": target_seam_stats,
        "pred_seams": pred_seam_stats,
        "error_seams": err_seam_stats,
    }

    if K is not None and E is not None:
        with torch.no_grad():
            target_camera = model.virtual_projector.render_cylinder_to_cameras(
                target, K, E, intr_hw,
                target_hw=(int(x.shape[-2]), int(x.shape[-1])),
            )
            target_coverage = model.virtual_projector.render_cylinder_to_cameras(
                torch.ones_like(target[:, :, :, :, :1]), K, E, intr_hw,
                target_hw=(int(x.shape[-2]), int(x.shape[-1])),
            )
        save_video_view_grid(
            out_dir / "projector_roundtrip" / "target_render_camera.png",
            target_camera.detach().cpu(),
            max_frames=max_frames,
            max_views=max_views,
            value_range="pm1",
        )
        roundtrip_err = (target_camera.detach().float() - x.detach().float()).abs()
        save_video_view_grid(
            out_dir / "projector_roundtrip" / "target_render_abs_error.png",
            roundtrip_err.cpu().clamp(0, 1),
            max_frames=max_frames,
            max_views=max_views,
            value_range="01",
        )
        coverage = target_coverage[0, :, :, 0].detach().float().cpu()
        save_tensor_heat_grid(
            out_dir / "projector_roundtrip" / "camera_render_coverage.png",
            coverage,
            max_frames=max_frames,
            max_views=max_views,
            fixed01=True,
        )
        stats["projector_roundtrip"] = {
            "l1": float(roundtrip_err.mean().item()),
            "coverage": tensor_stats(target_coverage),
            "coverage_zero_fraction": float(
                (target_coverage < 1e-4).float().mean().item()),
        }
    return stats


def split_latent_package(x: torch.Tensor,
                         model: CrossView4DVAE) -> list[tuple[str, torch.Tensor]]:
    counts = list(getattr(
        model.config, "layer_latent_token_counts",
        getattr(model.config, "layer_latent_view_counts", [])))
    labels = layer_labels(model)
    if not counts:
        return [("latent", x)]
    chunks = []
    start = 0
    for idx, count in enumerate(counts):
        label = labels[idx] if idx < len(labels) else f"layer{idx:02d}"
        chunks.append((label, x[:, :, start:start + int(count)]))
        start += int(count)
    return chunks


def latent_tensor_debug(out_dir: Path, prefix: str, tensor: torch.Tensor,
                        model: CrossView4DVAE, max_frames: int,
                        max_views: int) -> dict[str, Any]:
    stats = tensor_stats(tensor)
    if tensor.ndim != 6:
        return stats
    energy = tensor.detach().float().abs().mean(dim=3)[0]
    save_tensor_heat_grid(
        out_dir / "latent" / f"{prefix}_energy_all.png",
        energy,
        max_frames=max_frames,
        max_views=max_views,
    )
    per_layer = {}
    tv_shapes = list(getattr(model.config, "layer_latent_shapes", []))
    spatial_shapes = list(getattr(
        model.config, "layer_latent_spatial_shapes", []) or [])
    for idx, (label, part) in enumerate(split_latent_package(tensor, model)):
        part_energy = part.detach().float().abs().mean(dim=3)[0]
        save_tensor_heat_grid(
            out_dir / "latent" / f"{prefix}_energy_{label}.png",
            part_energy,
            max_frames=max_frames,
            max_views=max_views,
        )
        layer_stats = tensor_stats(part)
        if (
            part.ndim == 6
            and part.shape[-2:] == (1, 1)
            and idx < len(tv_shapes)
            and idx < len(spatial_shapes)
        ):
            t_count, v_count = [int(x) for x in tv_shapes[idx]]
            h_lat, w_lat = [int(x) for x in spatial_shapes[idx]]
            expected = t_count * v_count * h_lat * w_lat
            if part.shape[2] == expected:
                grid = part[0, 0, :, :, 0, 0].detach().float().abs().mean(
                    dim=-1).reshape(t_count, v_count, h_lat, w_lat)
                save_tensor_heat_grid(
                    out_dir / "latent" / f"{prefix}_spatial_{label}.png",
                    grid,
                    max_frames=max_frames,
                    max_views=max_views,
                )
                layer_stats["latent_time_view_shape"] = [t_count, v_count]
                layer_stats["latent_spatial_shape"] = [h_lat, w_lat]
        per_layer[label] = layer_stats
    stats["per_layer"] = per_layer
    return stats


def latent_debug(out_dir: Path, model: CrossView4DVAE,
                 activations: dict[str, torch.Tensor],
                 output: dict[str, Any], max_frames: int,
                 max_views: int) -> dict[str, Any]:
    shapes = list(getattr(model.config, "layer_latent_shapes", []))
    spatial_factors = list(getattr(
        model.config, "layer_latent_spatial_factors", []))
    spatial_shapes = list(getattr(
        model.config, "layer_latent_spatial_shapes", []) or [])
    token_counts = list(getattr(model.config, "layer_latent_token_counts", []))
    tv_token_counts = [
        int(t) * int(v) for t, v in shapes
    ] if shapes else []
    virtual_views = int(model.config.virtual_view_count)
    layer_count = max(len(shapes), 1)
    source_hw = list(getattr(model, "_last_bottleneck_hw", []) or [])
    source_spatial_tokens = (
        int(source_hw[0]) * int(source_hw[1]) if len(source_hw) == 2 else None)
    source_total = (
        int(getattr(model.config, "sequence_length", 0))
        * layer_count * virtual_views
        * source_spatial_tokens
        if source_spatial_tokens is not None else None
    )
    budget = {
        "source_time_tokens": int(getattr(model.config, "sequence_length", 0)),
        "virtual_views_per_layer": virtual_views,
        "layer_latent_time_view_counts": shapes,
        "layer_latent_spatial_downsample_factors": spatial_factors,
        "source_latent_spatial_shape": source_hw,
        "layer_latent_spatial_shapes": spatial_shapes,
        "layer_latent_time_view_token_counts": tv_token_counts,
        "layer_latent_token_counts": token_counts,
        "total_source_time_view_tokens": (
            int(getattr(model.config, "sequence_length", 0))
            * layer_count * virtual_views
        ),
        "total_latent_time_view_tokens": (
            sum(tv_token_counts) if tv_token_counts else None),
        "compression_ratio_time_view_axis": (
            (
                int(getattr(model.config, "sequence_length", 0))
                * layer_count * virtual_views
            ) / max(sum(tv_token_counts), 1)
            if tv_token_counts else None
        ),
        "total_source_tokens_including_spatial": source_total,
        "total_latent_tokens_including_spatial": (
            sum(token_counts) if token_counts else None),
        "compression_ratio_total_token_axis": (
            source_total / max(sum(token_counts), 1)
            if source_total is not None and token_counts else None
        ),
    }
    stats: dict[str, Any] = {"budget": budget}
    down = activations.get("view_down")
    if down is not None:
        stats["view_down"] = latent_tensor_debug(
            out_dir, "view_down", down, model, max_frames, max_views)
    stats["posterior_mean"] = latent_tensor_debug(
        out_dir, "posterior_mean", output["posterior"].mean.detach().cpu(),
        model, max_frames, max_views)
    stats["posterior_logvar"] = latent_tensor_debug(
        out_dir, "posterior_logvar", output["posterior"].logvar.detach().cpu(),
        model, max_frames, max_views)
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

    z_views = int(getattr(
        model.config, "total_latent_view_count",
        model.config.latent_view_count))
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
        "- `reconstruction/shared_camera_grid.png`: camera rendering from the shared layered cylinder path.",
        "- `reconstruction/abs_error.png`: absolute reconstruction error.",
        "- `cylinder/target_cylinder_grid*.png`: camera input projected into each cylinder layer.",
        "- `cylinder/pred_cylinder_grid*.png`: decoded near/far cylinder layers before camera rendering.",
        "- `cylinder/abs_error*.png`: per-layer cylinder-domain reconstruction error.",
        "- `cylinder/valid_mask*.png`: valid camera-to-cylinder supervision region per layer.",
        "- `cylinder/*_seams*.png`: right-edge to next-left-edge seam magnitude per layer/view.",
        "- `projector_roundtrip/target_render_camera.png`: projector-only camera->layered-cylinder->camera loop.",
        "- `projector_roundtrip/target_render_abs_error.png`: geometry-only roundtrip error, independent of VAE.",
        "- `latent/*_energy_*.png`: near/far latent package energy after T-V joint compression.",
        "- `activations/*_energy.png`: mean absolute feature energy by stage.",
        "- `activations/*_view_cosine.png`: cross-view feature cosine matrix.",
        "- `projector/*_coverage*.png`: how much source-camera signal reaches each virtual pixel/layer.",
        "- `projector/source_entropy.png`: whether projector blends many cameras or picks one.",
        "- `projector/source_camera_rgb.png`: dominant source-camera color map.",
        "- `projector/fallback_projected_rgb.png`: projector output with camera params disabled.",
        "- `projector/camera_projected_rgb*.png`: layered projector output with K/E enabled, if available.",
        "- `projector/camera_projected_mask*.png`: valid supervision mask for cylinder-domain loss.",
        "- `projector/camera_vs_fallback_absdiff.png`: how much K/E changes the virtual rig.",
        "- `projector/*_coverage.png` and `projector/*_source_camera_rgb.png`: fallback/active/camera source routing.",
        "- `rope/*_nearest_distance.png`: nearest RoPE query/key coordinate distance.",
        "- `rope/*_query_azimuth.png`: query azimuth coordinate layout.",
        "",
        "## How To Read It",
        "",
        "- Input bug: blank/dark columns in `input_grid.png`, view means near zero, or huge random `temporal_delta` on synthetic data.",
        "- Projector bug: `coverage_zero_fraction` high, all-black coverage, or source colors not matching expected camera order. Check this separately for near/far layers.",
        "- Renderer / cylinder representation bug: `projector_roundtrip/target_render_abs_error.png` has gray bands, vertical seams, or high `projector_roundtrip.l1` before the VAE is involved.",
        "- Cylinder target bug: `cylinder/target_seams*.png` has strong stripes. That means the camera->cylinder target itself is discontinuous.",
        "- Decoder seam bug: target seams are weak but `cylinder/pred_seams*.png` or `cylinder/error_seams*.png` is strong.",
        "- Layer-budget issue: `latent/budget` shows near/far token counts. If far is clean but near is noisy or collapsed, near compression is too strong; if both are flat, latent channels or decoder capacity are too small.",
        "- Coordinate-convention bug: `debug_metrics.json -> projector -> camera_geometry` shows camera azimuth in both nuScenes xy-ground and old xz-ground conventions. If only the old convention looks coherent, synthetic assumptions leaked into real data; if only xy looks coherent, the projector must use nuScenes axes.",
        "- Closed-loop camera loss: with this script, `--use-camera-params` also renders decoded cylinder output back to physical cameras before comparing to the raw input.",
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
    if args.decode_camera_from_cylinder and not args.use_camera_params:
        args.use_camera_params = True
    args.decode_camera_from_cylinder = args.use_camera_params
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
    if (
        virtual_views != v_input
        and not args.use_camera_params
    ):
        raise ValueError(
            "visual_debug expects virtual_view_count == input view count so "
            "reconstruction can be compared directly. For canonical cylinder "
            "debug with a different virtual view count, pass "
            "--use-camera-params. Got "
            f"{virtual_views} vs {v_input}."
        )
    latent_views = args.latent_view_count if args.latent_view_count > 0 else v_input
    if latent_views > virtual_views:
        raise ValueError("latent_view_count must be <= virtual_view_count")
    near_views = (
        args.near_latent_view_count
        if args.near_latent_view_count > 0 else latent_views)
    far_views = (
        args.far_latent_view_count
        if args.far_latent_view_count > 0 else max(1, latent_views // 2))
    near_times = (
        args.near_latent_time_count
        if args.near_latent_time_count > 0 else args.latent_time_count)
    far_times = (
        args.far_latent_time_count
        if args.far_latent_time_count > 0
        else max(1, args.latent_time_count // 2))
    if args.latent_spatial_downsample_factor < 1:
        raise ValueError("latent_spatial_downsample_factor must be >= 1")
    near_spatial = (
        args.near_latent_spatial_downsample_factor
        if args.near_latent_spatial_downsample_factor > 0
        else args.latent_spatial_downsample_factor)
    far_spatial = (
        args.far_latent_spatial_downsample_factor
        if args.far_latent_spatial_downsample_factor > 0
        else max(1, args.latent_spatial_downsample_factor * 2))
    if near_views > virtual_views or far_views > virtual_views:
        raise ValueError("near/far latent view counts must be <= virtual_view_count")
    if near_times > args.sequence_length or far_times > args.sequence_length:
        raise ValueError("near/far latent time counts must be <= sequence_length")

    model = CrossView4DVAE(
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        sequence_length=args.sequence_length,
        virtual_view_count=virtual_views,
        latent_view_count=latent_views,
        latent_time_count=args.latent_time_count,
        image_height=args.image_hw[0],
        image_width=args.image_hw[1],
        near_latent_time_count=near_times,
        far_latent_time_count=far_times,
        near_latent_view_count=near_views,
        far_latent_view_count=far_views,
        latent_spatial_downsample_factor=args.latent_spatial_downsample_factor,
        near_latent_spatial_downsample_factor=near_spatial,
        far_latent_spatial_downsample_factor=far_spatial,
        num_attention_heads=args.num_attention_heads,
        num_bottleneck_blocks=args.num_bottleneck_blocks,
        temporal_downsample_factor=args.temporal_downsample_factor,
        temporal_pre=args.temporal_pre,
        spatial_downsample_factor=args.spatial_downsample_factor,
        cylinder_radii=tuple(args.cylinder_radii),
        cylinder_height_scale=args.cylinder_height_scale,
        cylinder_vertical_mode=args.cylinder_vertical_mode,
        projector_edge_feather=args.projector_edge_feather,
        projector_angular_power=args.projector_angular_power,
        projector_blend_mode=args.projector_blend_mode,
        ego_coordinate_mode=args.ego_coordinate_mode,
        decode_camera_from_cylinder=args.decode_camera_from_cylinder,
        use_view_residual=False,
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
        value_range="pm1",
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
        value_range="pm1",
    )
    shared_sample = output.get("shared_sample")
    if shared_sample is not None and shared_sample.shape == x.shape:
        save_video_view_grid(
            out_dir / "reconstruction" / "shared_camera_grid.png",
            shared_sample.detach().cpu(),
            max_frames=args.max_frames,
            max_views=args.max_views,
            value_range="pm1",
        )
    if "cylinder_sample" in output:
        save_layered_video_grids(
            out_dir / "reconstruction" / "cylinder_recon_grid.png",
            output["cylinder_sample"].detach().cpu(),
            model,
            max_frames=args.max_frames,
            max_views=args.max_views,
            value_range="pm1",
        )
    depth_mask = output.get("depth_routing_mask")
    if depth_mask is not None:
        save_tensor_heat_grid(
            out_dir / "latent" / "depth_routing_near_mask.png",
            depth_mask[0, :, :, 0].detach().float().cpu(),
            max_frames=args.max_frames,
            max_views=args.max_views,
        )
    save_video_view_grid(
        out_dir / "reconstruction" / "abs_error.png",
        err.detach().cpu().clamp(0, 1),
        max_frames=args.max_frames,
        max_views=args.max_views,
        value_range="01",
    )

    metrics: dict[str, Any] = {
        "input": tensor_stats(x),
        "reconstruction_l1": float(F.l1_loss(recon, x).item()),
        "reconstruction_mse": float(F.mse_loss(recon, x).item()),
        "shared_reconstruction": tensor_stats(shared_sample)
        if shared_sample is not None else None,
        "posterior_mean": tensor_stats(output["posterior"].mean),
        "posterior_logvar": tensor_stats(output["posterior"].logvar),
        "depth_routing_mask": (
            tensor_stats(depth_mask) if depth_mask is not None else None),
    }

    metrics["cylinder"] = cylinder_debug(
        model, x, output, K, E, intr_hw, out_dir,
        args.max_frames, args.max_views)

    if K is not None and E is not None and "cylinder_sample" in output:
        cyl = output["cylinder_sample"]
        with torch.no_grad():
            if cyl.ndim == 7:
                ones_cyl = torch.ones_like(cyl[:, :, :, :, :1])
            else:
                ones_cyl = torch.ones_like(cyl[:, :, :, :1])
            render_coverage = model.virtual_projector.render_cylinder_to_cameras(
                ones_cyl,
                K,
                E,
                intr_hw,
                target_hw=(int(x.shape[-2]), int(x.shape[-1])),
            )
        coverage = render_coverage[0, :, :, 0].detach().cpu()
        save_tensor_heat_grid(
            out_dir / "reconstruction" / "camera_render_coverage.png",
            coverage,
            max_frames=args.max_frames,
            max_views=args.max_views,
            fixed01=True,
        )
        metrics["camera_render_coverage"] = {
            "shape": list(render_coverage.shape),
            "mean": float(render_coverage.mean().item()),
            "zero_fraction": float(
                (render_coverage < 1e-4).float().mean().item()),
        }

    activation_stats = {}
    for name, tensor in activations.items():
        activation_stats[name] = save_activation_debug(
            out_dir, name, tensor, args.max_frames, args.max_views)
    metrics["activations"] = activation_stats

    metrics["projector"] = projector_debug(
        model, batch, args.use_camera_params, out_dir,
        args.max_frames, args.max_views, device)
    metrics["latent"] = latent_debug(
        out_dir, model, activations, output,
        args.max_frames, args.max_views)
    metrics["rope"] = rope_debug(out_dir, activations, model, args.max_views)

    write_json(out_dir / "debug_metrics.json", metrics)
    write_report(out_dir, args, metrics)

    print(f"[done] wrote visual debug report to {out_dir}")
    print(f"[done] open {out_dir / 'report.md'}")


if __name__ == "__main__":
    # Local Windows environments with both MKL and torch OpenMP sometimes need:
    # $env:KMP_DUPLICATE_LIB_OK='TRUE'
    main()
