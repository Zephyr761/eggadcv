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

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T_
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
            pano = self._draw_panorama(t, sample_seed + t)
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

    def _stack_pil(self, pil_nested) -> torch.Tensor:
        H, W = self.image_hw
        T_len = min(len(pil_nested), self.T)
        V = len(pil_nested[0]) if T_len else 0
        out = torch.zeros(T_len, V, 3, H, W)
        for t in range(T_len):
            for v in range(V):
                im = pil_nested[t][v]
                if isinstance(im, Image.Image):
                    out[t, v] = self._tx(im)
                # missing frame -> zeros
        return out

    def __getitem__(self, idx):
        sample = self.base[idx]
        pil = sample.get("images") or sample.get("vae_images")
        if pil is None:
            raise KeyError(
                "Sample has neither 'images' nor 'vae_images'. Found keys: "
                + str(list(sample.keys())))
        imgs = self._stack_pil(pil)                  # [T, V, 3, H, W] in [0, 1]
        if self.normalize_to_pm1:
            imgs = imgs * 2 - 1

        K = sample.get("camera_intrinsics")
        if K is not None and K.dim() == 4 and K.shape[-1] == 4:
            K = K[..., :3, :3]
        if K is not None:
            K = K[: imgs.shape[0]].float()           # [T, V, 3, 3]
        E = sample.get("camera_transforms")
        if E is not None:
            E = E[: imgs.shape[0]].float()           # [T, V, 4, 4]

        # Recover original calibration resolution from ``image_size`` so the
        # projector can rescale intrinsics. Fallback to the resized HW if
        # the field is missing (then intrinsics are assumed to live at the
        # current resolution).
        orig_hw = self.image_hw
        sz = sample.get("image_size")
        if torch.is_tensor(sz) and sz.numel() >= 2:
            # ``image_size`` is stored as (W, H) per the dataset README.
            w0 = int(sz[0, 0, 0]); h0 = int(sz[0, 0, 1])
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
