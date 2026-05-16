"""Datasets for the cross-view 4D VAE training.

Two flavours:

1. :class:`SyntheticCylinderDataset` -- a tiny self-contained dataset that
   renders coloured stripes around a cylinder so different cameras genuinely
   share a panorama. Use this to overfit and verify the model wiring before
   touching real data.

2. :class:`NuPlanVAEAdapter` -- wraps the existing
   ``dwm.datasets.nuplan.NuPlanDataset`` (which returns nested PIL lists) and
   produces the ``[T, V, C, H, W]`` tensor layout the VAE expects, plus
   intrinsics / extrinsics that the geometry-aware projector can consume.

Both honour the temporal-pre constraint of the VAE:
``T = temporal_pre + k * temporal_downsample_factor``
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T_
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_SENSOR_CHANNELS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
]


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
@dataclass
class SyntheticConfig:
    samples_per_epoch: int = 256
    sequence_length: int = 5         # T
    view_count: int = 6              # V
    image_hw: tuple = (64, 128)      # (H, W) -> 2:1 aspect ratio per view
    seed: int = 0


class SyntheticCylinderDataset(Dataset):
    """Generates synthetic 360-deg cylinder footage with V cameras.

    Each ego frame draws coloured vertical stripes around a virtual cylinder.
    Each input camera takes a 60-deg slice of that cylinder + a tiny per-view
    perturbation, so the VAE can actually learn the cross-view alignment.

    Output layout per sample:
        ``vae_images``: ``[T, V, 3, H, W]`` in ``[-1, 1]``
        ``camera_intrinsics``: ``[T, V, 3, 3]``
        ``camera_transforms``: ``[T, V, 4, 4]`` (camera_to_ego)
        ``intrinsics_hw``: tuple ``(H, W)`` (calibration resolution)
    """

    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)

    def __len__(self):
        return self.cfg.samples_per_epoch

    def _draw_panorama(self, t: int, frame_seed: int) -> torch.Tensor:
        """Make a 360-deg panorama at time ``t``. Shape ``[3, H, W_pano]``."""
        H, W = self.cfg.image_hw
        W_pano = W * self.cfg.view_count
        # Generate vertical colour stripes that drift over time so consecutive
        # frames are correlated (gives temporal causality something to learn).
        rng = torch.Generator().manual_seed(frame_seed)
        n_stripes = 16
        # Per-stripe colour (constant within a sample), per-stripe x-position
        # at t=0 (constant), shared horizontal drift speed.
        colours = torch.rand(n_stripes, 3, generator=rng)
        x0 = torch.rand(n_stripes, generator=rng) * W_pano
        speed = (torch.rand(n_stripes, generator=rng) - 0.5) * (W_pano / 16)
        x_t = (x0 + speed * t) % W_pano
        widths = 4 + (torch.rand(n_stripes, generator=rng) * 12).int()

        pano = torch.zeros(3, H, W_pano)
        # Sky gradient (top half lighter).
        v = torch.linspace(0.4, 0.1, H).view(1, H, 1)
        pano = pano + v
        # Stripes.
        for i in range(n_stripes):
            cx = int(x_t[i].item())
            half = max(int(widths[i].item()) // 2, 1)
            for k in range(-half, half + 1):
                col = (cx + k) % W_pano
                pano[:, :, col] = colours[i].view(3, 1)
        # Tiny noise.
        pano = pano + 0.02 * torch.randn(pano.shape, generator=rng)
        return pano.clamp(0, 1)

    def _slice_views(self, pano: torch.Tensor) -> torch.Tensor:
        """Cut V evenly-spaced ``W``-wide windows from the ``W*V`` panorama."""
        H, W = self.cfg.image_hw
        V = self.cfg.view_count
        out = torch.empty(V, 3, H, W)
        for v in range(V):
            out[v] = pano[:, :, v * W: (v + 1) * W]
        return out

    def _camera_params(self) -> tuple:
        """Generate K (3x3) and ego2cam-compatible extrinsics for each view."""
        H, W = self.cfg.image_hw
        V = self.cfg.view_count
        # Each view covers 360/V degrees, so f = (W/2) / tan(fov/2).
        fov = 2 * math.pi / V
        fx = (W / 2) / math.tan(fov / 2)
        fy = fx  # square pixels
        cx, cy = W / 2.0, H / 2.0
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        Ks = K.unsqueeze(0).repeat(V, 1, 1)

        # Camera_to_ego: rotated around y-axis, with 1m radius offset
        # (matches typical AV mounting baseline scale).
        exts = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
        for v in range(V):
            ang = 2 * math.pi * v / V
            R = torch.tensor([
                [math.cos(ang), 0, math.sin(ang)],
                [0, 1, 0],
                [-math.sin(ang), 0, math.cos(ang)],
            ], dtype=torch.float32)
            exts[v, :3, :3] = R
            exts[v, :3, 3] = torch.tensor([math.sin(ang), 0, math.cos(ang)])
        return Ks, exts

    def __getitem__(self, idx):
        H, W = self.cfg.image_hw
        T_len, V = self.cfg.sequence_length, self.cfg.view_count
        sample_seed = self.cfg.seed + idx * 9173
        frames = []
        for t in range(T_len):
            pano = self._draw_panorama(t, sample_seed)
            views = self._slice_views(pano)
            frames.append(views)
        imgs = torch.stack(frames, dim=0)            # [T, V, 3, H, W]
        imgs = imgs * 2 - 1                          # [-1, 1]

        Ks, exts = self._camera_params()
        Ks_t = Ks.unsqueeze(0).expand(T_len, V, 3, 3).clone()
        exts_t = exts.unsqueeze(0).expand(T_len, V, 4, 4).clone()

        return {
            "vae_images": imgs,
            "camera_intrinsics": Ks_t,
            "camera_transforms": exts_t,
            "intrinsics_hw": (H, W),
        }


# ---------------------------------------------------------------------------
# Multi-view dataset adapter (works with nuPlan, nuScenes, Waymo, ...)
# ---------------------------------------------------------------------------
class MultiViewVAEAdapter(Dataset):
    """Wrap any ``dwm.datasets.*MotionDataset`` (nuScenes / nuPlan / Waymo /
    Argoverse) and produce the tensor layout the VAE expects.

    All those base datasets follow the same convention documented in
    ``src/dwm/datasets/README.md``:

    - ``images``: nested PIL list ``[T][V]`` at the *original* image size.
    - ``camera_intrinsics``: float tensor ``[T, V, 3, 3]``.
    - ``camera_transforms``: float tensor ``[T, V, 4, 4]`` (camera -> ego).
    - ``image_size``: long tensor ``[T, V, 2]`` (W, H).

    The adapter:
      * resizes every PIL frame to ``image_hw`` and stacks to a
        ``[T, V, 3, H, W]`` tensor in ``[-1, 1]``;
      * keeps the **original** intrinsics calibration resolution (so the
        VAE projector can rescale internally).

    Args:
        base_dataset: an instance returning samples in the layout above.
        sequence_length: number of frames T to keep. Must satisfy the VAE's
            ``T = temporal_pre + k * temporal_downsample_factor`` constraint.
        image_hw: target resize ``(H, W)``.
    """

    def __init__(
        self,
        base_dataset,
        sequence_length: int,
        image_hw: tuple = (256, 448),
        normalize_to_pm1: bool = True,
    ):
        self.base = base_dataset
        self.T = sequence_length
        self.image_hw = image_hw
        self.normalize_to_pm1 = normalize_to_pm1
        self._tx = T_.Compose([
            T_.Resize(image_hw),
            T_.ToTensor(),
        ])

    def __len__(self):
        return len(self.base)

    def _image_to_tensor(self, im) -> torch.Tensor:
        if isinstance(im, Image.Image):
            return self._tx(im)

        if torch.is_tensor(im):
            x = im.detach().cpu()
        else:
            try:
                x = torch.as_tensor(im)
            except Exception as exc:
                raise TypeError(
                    "Unsupported image type in MultiViewVAEAdapter: "
                    f"{type(im).__name__}. Expected PIL, tensor, or numpy-like "
                    "array."
                ) from exc

        if x.ndim != 3:
            raise ValueError(
                "Unsupported image tensor shape in MultiViewVAEAdapter: "
                f"{tuple(x.shape)}. Expected [C,H,W] or [H,W,C]."
            )
        if x.shape[0] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
            x = x.permute(2, 0, 1)
        if x.shape[0] not in (1, 3, 4):
            raise ValueError(
                "Unsupported image tensor channel count in MultiViewVAEAdapter: "
                f"{tuple(x.shape)}."
            )

        x = x.float()
        if x.max().item() > 2.0:
            x = x / 255.0
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        elif x.shape[0] == 4:
            x = x[:3]

        x = F.interpolate(
            x.unsqueeze(0),
            size=self.image_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return x.clamp(0, 1)

    def _resize_chw_video(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, V, C, H, W], value range may be [0,1], [0,255], or [-1,1].
        if x.ndim != 5:
            raise ValueError(
                "Expected video tensor [T,V,C,H,W], got "
                f"{tuple(x.shape)}."
            )
        t, v, c, h, w = x.shape
        if c not in (1, 3, 4):
            raise ValueError(
                "Expected video channel count 1/3/4 at dim 2, got "
                f"{tuple(x.shape)}."
            )
        x = x.float()
        if c == 1:
            x = x.expand(t, v, 3, h, w)
        elif c == 4:
            x = x[:, :, :3]
        if (h, w) != tuple(self.image_hw):
            x = F.interpolate(
                x.reshape(t * v, 3, h, w),
                size=self.image_hw,
                mode="bilinear",
                align_corners=False,
            ).reshape(t, v, 3, self.image_hw[0], self.image_hw[1])
        return x

    def _stack_video_tensor(self, video) -> torch.Tensor:
        x = video.detach().cpu() if torch.is_tensor(video) else torch.as_tensor(video)
        if x.ndim == 6:
            if x.shape[0] != 1:
                raise ValueError(
                    "MultiViewVAEAdapter received a batched video tensor with "
                    f"shape {tuple(x.shape)}. A dataset __getitem__ should "
                    "return one clip [T,V,C,H,W]; remove the leading batch "
                    "dimension before wrapping it."
                )
            x = x[0]
        if x.ndim != 5:
            raise ValueError(
                "Unsupported vae_images tensor shape: "
                f"{tuple(x.shape)}. Expected [T,V,C,H,W]."
            )
        if x.shape[2] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 1, 4, 2, 3)
        x = x[: self.T]
        x = self._resize_chw_video(x)
        # Direct ``vae_images`` tensors may already be in [-1, 1]. Return the
        # adapter's canonical range here so callers do not double-normalize.
        if x.numel() and x.min().item() < -0.05:
            return x.clamp(-1, 1)
        if x.numel() and x.max().item() > 2.0:
            x = x / 255.0
        x = x.clamp(0, 1)
        return x * 2 - 1 if self.normalize_to_pm1 else x

    @staticmethod
    def _strip_single_batch(x, expected_unbatched_ndim: int):
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.ndim == expected_unbatched_ndim + 1:
            if x.shape[0] != 1:
                raise ValueError(
                    "Expected a single sample, but got a batched tensor with "
                    f"shape {tuple(x.shape)}."
                )
            x = x[0]
        return x

    def _stack_frame_list(self, pil_nested) -> torch.Tensor:
        H, W = self.image_hw
        T_len = min(len(pil_nested), self.T)
        V = len(pil_nested[0]) if T_len else 0
        out = torch.zeros(T_len, V, 3, H, W)
        for t in range(T_len):
            for v in range(V):
                im = pil_nested[t][v]
                if im is None:
                    # missing frame -> zeros
                    continue
                out[t, v] = self._image_to_tensor(im)
        return out

    def __getitem__(self, idx):
        sample = self.base[idx]
        if not isinstance(sample, dict):
            imgs = self._stack_video_tensor(sample)
            return {
                "vae_images": imgs,
                "camera_intrinsics": None,
                "camera_transforms": None,
                "intrinsics_hw": self.image_hw,
            }

        if "images" in sample and sample["images"] is not None:
            raw_images = sample["images"]
            imgs = self._stack_frame_list(raw_images)  # [T,V,3,H,W] in [0,1]
            if self.normalize_to_pm1:
                imgs = imgs * 2 - 1
        elif "vae_images" in sample and sample["vae_images"] is not None:
            imgs = self._stack_video_tensor(sample["vae_images"])
        else:
            raise KeyError(
                "Sample has neither 'images' nor 'vae_images'. Found keys: "
                + str(list(sample.keys())))

        K = sample.get("camera_intrinsics")
        K = self._strip_single_batch(K, expected_unbatched_ndim=4)
        if K is not None and K.dim() == 4 and K.shape[-1] == 4:
            K = K[..., :3, :3]
        if K is not None:
            K = K[: imgs.shape[0]].float()           # [T, V, 3, 3]
        E = sample.get("camera_transforms")
        E = self._strip_single_batch(E, expected_unbatched_ndim=4)
        if E is not None:
            E = E[: imgs.shape[0]].float()           # [T, V, 4, 4]

        # Recover original calibration resolution from ``image_size`` so the
        # projector can rescale intrinsics. Fallback to the resized HW if
        # the field is missing (then intrinsics are assumed to live at the
        # current resolution).
        orig_hw = self.image_hw
        sz = sample.get("image_size")
        sz = self._strip_single_batch(sz, expected_unbatched_ndim=3)
        if torch.is_tensor(sz) and sz.numel() >= 2:
            # ``image_size`` is stored as (W, H) per the dataset README.
            w0 = int(sz[0, 0, 0])
            h0 = int(sz[0, 0, 1])
            if h0 > 0 and w0 > 0:
                orig_hw = (h0, w0)

        return {
            "vae_images": imgs,
            "camera_intrinsics": K,
            "camera_transforms": E,
            "intrinsics_hw": orig_hw,
        }


# Back-compat alias.
NuPlanVAEAdapter = MultiViewVAEAdapter


# ---------------------------------------------------------------------------
# Scene-folder nuScenes export adapter
# ---------------------------------------------------------------------------
class NuScenesSceneFolderDataset(Dataset):
    """Load scene-wise nuScenes exports with ``images/`` + ``opencv_cameras.json``.

    Expected layout:

    ``root/<split>/scene-xxxx/images/*.jpg``
    ``root/<split>/scene-xxxx/opencv_cameras.json``

    The JSON is expected to contain a flat ``frames`` list where each item
    describes a single camera image via fields such as ``file_path``, ``fx``,
    ``fy``, ``cx``, ``cy`` and ``w2c``.
    """

    frame_re = re.compile(r"(?P<scene>.+)_(?P<camera>CAM_[A-Z_]+)_(?P<frame>\d+)\.(jpg|png)$")

    def __init__(
        self,
        root: str,
        split: str,
        sequence_length: int,
        sensor_channels: Optional[Sequence[str]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.sequence_length = sequence_length
        self.sensor_channels = list(sensor_channels or DEFAULT_SENSOR_CHANNELS)
        self.scene_root = self.root / split
        if not self.scene_root.is_dir():
            raise FileNotFoundError(
                f"Scene-folder split directory not found: {self.scene_root}")

        self.scenes = []
        self.sample_index = []
        scene_dirs = sorted(p for p in self.scene_root.iterdir() if p.is_dir())
        if not scene_dirs:
            raise RuntimeError(f"No scene directories found under {self.scene_root}")

        for scene_dir in scene_dirs:
            camera_json = scene_dir / "opencv_cameras.json"
            if not camera_json.is_file():
                continue
            with open(camera_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            grouped = self._group_frames(scene_dir, meta.get("frames", []))
            if len(grouped) < self.sequence_length:
                continue
            scene_idx = len(self.scenes)
            self.scenes.append({
                "scene_dir": scene_dir,
                "frames": grouped,
                "scene_name": meta.get("scene_name", scene_dir.name),
            })
            for start in range(len(grouped) - self.sequence_length + 1):
                self.sample_index.append((scene_idx, start))

        if not self.sample_index:
            raise RuntimeError(
                f"No valid clips of length {self.sequence_length} found under "
                f"{self.scene_root}")

    def _group_frames(self, scene_dir: Path, entries: Sequence[dict]) -> list:
        grouped = {}
        for item in entries:
            file_path = item.get("file_path", "")
            match = self.frame_re.match(os.path.basename(file_path))
            if match is None:
                continue
            camera = match.group("camera")
            frame_idx = int(match.group("frame"))
            per_frame = grouped.setdefault(frame_idx, {})
            per_frame[camera] = {
                "path": scene_dir / file_path,
                "fx": float(item["fx"]),
                "fy": float(item["fy"]),
                "cx": float(item["cx"]),
                "cy": float(item["cy"]),
                "w": int(item["w"]),
                "h": int(item["h"]),
                "w2c": torch.tensor(item["w2c"], dtype=torch.float32),
                "timestamp": int(item.get("timestamp", frame_idx)),
            }

        result = []
        for frame_idx in sorted(grouped):
            per_frame = grouped[frame_idx]
            if not all(cam in per_frame for cam in self.sensor_channels):
                continue
            result.append({
                "frame_idx": frame_idx,
                "timestamp": min(per_frame[cam]["timestamp"] for cam in self.sensor_channels),
                "cameras": per_frame,
            })
        return result

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        scene_idx, start = self.sample_index[idx]
        scene = self.scenes[scene_idx]
        clip = scene["frames"][start:start + self.sequence_length]

        images = []
        K_list = []
        E_list = []
        image_size = []

        for frame in clip:
            frame_images = []
            frame_K = []
            frame_E = []
            frame_size = []
            for cam in self.sensor_channels:
                cam_info = frame["cameras"][cam]
                with Image.open(cam_info["path"]) as im:
                    frame_images.append(im.convert("RGB"))
                frame_K.append(torch.tensor([
                    [cam_info["fx"], 0.0, cam_info["cx"]],
                    [0.0, cam_info["fy"], cam_info["cy"]],
                    [0.0, 0.0, 1.0],
                ], dtype=torch.float32))
                frame_E.append(torch.linalg.inv(cam_info["w2c"]))
                frame_size.append(torch.tensor([cam_info["w"], cam_info["h"]], dtype=torch.long))
            images.append(frame_images)
            K_list.append(torch.stack(frame_K, dim=0))
            E_list.append(torch.stack(frame_E, dim=0))
            image_size.append(torch.stack(frame_size, dim=0))

        return {
            "images": images,
            "camera_intrinsics": torch.stack(K_list, dim=0),
            "camera_transforms": torch.stack(E_list, dim=0),
            "image_size": torch.stack(image_size, dim=0),
        }


def make_nuscenes_scene_folder_base(
    data_root: str,
    split: str,
    sequence_length: int,
    sensor_channels: Optional[Sequence[str]] = None,
):
    return NuScenesSceneFolderDataset(
        root=data_root,
        split=split,
        sequence_length=sequence_length,
        sensor_channels=sensor_channels,
    )


# ---------------------------------------------------------------------------
# nuScenes factory
# ---------------------------------------------------------------------------
def make_nuscenes_base(
    data_root: str,
    cache_root: Optional[str] = None,
    dataset_name: str = "v1.0-trainval",
    split: str = "train",
    sequence_length: int = 5,
    fps: int = 2,
    stride: float = 0.5,
    sensor_channels: Optional[Sequence[str]] = None,
    keyframe_only: bool = True,
):
    """Build a ``dwm.datasets.nuscenes.MotionDataset`` with sensible defaults.

    Args:
        data_root: filesystem path to the directory that contains the
            extracted nuScenes blobs (the same directory the ``nuscenes_fs``
            in the ctsd configs points at). On the lab machine in the
            project README this was ``.../dataset/nuscenes``.
        cache_root: optional path to a pre-generated image cache (greatly
            speeds up training). Pass ``None`` to read images straight from
            blobs.
        dataset_name: metadata sub-directory name. Common values:
            ``"v1.0-trainval"`` (annotated 2 Hz keyframes; 1.0 official) or
            ``"interp_12Hz_trainval"`` (interpolated 12 Hz frames -- needs
            the corresponding json metadata generated separately).
        split: ``"train"`` / ``"val"`` / ``"mini_train"`` / ``"mini_val"``.
        sequence_length: number of frames per clip.
        fps, stride: temporal sampling, see ``MotionDataset`` docs.
        sensor_channels: list of camera names. Defaults to the standard
            6-camera ring used in this project.
        keyframe_only: if ``True`` only keyframes (2 Hz) are used. Set to
            ``False`` only with the interpolated 12 Hz dataset.
    """
    import fsspec.implementations.dirfs as _dirfs
    import fsspec.implementations.local as _local
    from dwm.datasets.nuscenes import MotionDataset

    fs = _dirfs.DirFileSystem(path=data_root, fs=_local.LocalFileSystem())

    if sensor_channels is None:
        sensor_channels = [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
        ]

    kwargs = dict(
        fs=fs,
        dataset_name=dataset_name,
        sequence_length=sequence_length,
        fps_stride_tuples=[(fps, stride)],
        split=split,
        sensor_channels=list(sensor_channels),
        keyframe_only=keyframe_only,
        enable_synchronization_check=False,
        enable_camera_transforms=True,
    )
    if cache_root is not None:
        kwargs["cache_root"] = cache_root
    return MotionDataset(**kwargs)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def vae_collate(samples: Sequence[dict]) -> dict:
    """Stack a list of dict samples on the leading batch dim."""
    out = {
        "vae_images": torch.stack([s["vae_images"] for s in samples], dim=0),
    }
    K = [s.get("camera_intrinsics") for s in samples]
    if all(k is not None for k in K):
        out["camera_intrinsics"] = torch.stack(K, dim=0)
    E = [s.get("camera_transforms") for s in samples]
    if all(e is not None for e in E):
        out["camera_transforms"] = torch.stack(E, dim=0)
    # Same calibration resolution for the whole batch (assumed constant).
    if "intrinsics_hw" in samples[0]:
        out["intrinsics_hw"] = samples[0]["intrinsics_hw"]
    return out


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ds = SyntheticCylinderDataset(SyntheticConfig(samples_per_epoch=4))
    print("len:", len(ds))
    s = ds[0]
    for k, v in s.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)}  range=[{v.min():.2f}, {v.max():.2f}]")
        else:
            print(f"  {k}: {v}")
    batch = vae_collate([ds[i] for i in range(2)])
    print("\nbatched:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")
