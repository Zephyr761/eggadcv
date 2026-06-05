"""Camera-conditioned scene-token VAE with switchable T/scene compression.

No cylinder, BEV, or triplane representation is used here.  Camera parameters
only enter as ray/camera embeddings so the latent axis is a fixed scene-token
axis instead of a fixed camera-view axis.

Modes:

- ``tv_compression="joint"``: cross-attention compresses the T x scene-token
  grid jointly into T_lat x S_lat tokens.
- ``tv_compression="non_joint"``: T and scene-token axes are compressed by
  separable pooling/refinement, without T/scene cross-attention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.logvar = logvar.clamp(-30.0, 20.0)

    def sample(self) -> torch.Tensor:
        return self.mean + torch.randn_like(self.mean) * torch.exp(
            0.5 * self.logvar)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return -0.5 * torch.sum(
            1 + self.logvar - self.mean.pow(2) - self.logvar.exp())


@dataclass
class TVVAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 48
    latent_channels: int = 32
    sequence_length: int = 5
    view_count: int = 6
    scene_token_count: int = 8
    latent_time_count: int = 2
    latent_scene_token_count: int = 4
    spatial_downsample_factor: int = 4
    num_attention_heads: int = 4
    tv_compression: str = "joint"
    image_height: Optional[int] = None
    image_width: Optional[int] = None


class DecoderOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _per_frame_groupnorm(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    # x: [N,C,T,H,W]. Fold T into batch so normalization stays causal.
    n, c, t, h, w = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    y = norm(y)
    return y.view(n, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class CausalConv3d(nn.Conv3d):
    """Wan-style causal 3D convolution for [N,C,T,H,W]."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p_t, p_h, p_w = self.padding
        self._causal_padding = (int(p_w), int(p_h), int(2 * p_t), 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None):
        p_w, p_h, p_t_left, p_t_right = self._causal_padding
        if cache_x is not None and p_t_left > 0:
            cache_x = cache_x.to(device=x.device, dtype=x.dtype)
            x = torch.cat([cache_x, x], dim=2)
            p_t_left = max(p_t_left - cache_x.shape[2], 0)
        x = F.pad(x, (p_w, p_w, p_h, p_h, p_t_left, p_t_right))
        return super().forward(x)


class ResDownBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = CausalConv3d(
            channels, out_channels, 3, stride=(1, 2, 2), padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        self.skip = CausalConv3d(
            channels, out_channels, 1, stride=(1, 2, 2), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        y = self.conv1(F.silu(_per_frame_groupnorm(self.norm1, x)))
        y = self.conv2(F.silu(_per_frame_groupnorm(self.norm2, y)))
        return y + skip


class ResUpBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = CausalConv3d(channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, scale_factor=(1, 2, 2), mode="trilinear",
            align_corners=False)
        y = self.conv1(F.silu(_per_frame_groupnorm(self.norm1, x)))
        return self.conv2(F.silu(_per_frame_groupnorm(self.norm2, y)))


class GeometryEmbedding(nn.Module):
    """Create low-resolution ray/camera embeddings from K/E."""

    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    @staticmethod
    def _camera_features(
        b: int,
        t: int,
        v: int,
        h: int,
        w: int,
        dtype,
        device,
        intrinsics: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
        intrinsics_hw: Optional[tuple[int, int]],
    ) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype) + 0.5,
            torch.arange(w, device=device, dtype=dtype) + 0.5,
            indexing="ij")
        uv_norm = torch.stack(
            ((xx / max(w, 1)) * 2 - 1, (yy / max(h, 1)) * 2 - 1),
            dim=-1)
        uv_norm = uv_norm.view(1, 1, 1, h, w, 2).expand(b, t, v, h, w, 2)

        if intrinsics is None or extrinsics is None:
            ray = torch.zeros(b, t, v, h, w, 3, device=device, dtype=dtype)
            center = torch.zeros(b, t, v, h, w, 3, device=device, dtype=dtype)
            return torch.cat([uv_norm, ray, center], dim=-1)

        k = intrinsics.to(device=device, dtype=dtype)
        e = extrinsics.to(device=device, dtype=dtype)
        if k.ndim != 5 or e.ndim != 5:
            raise ValueError(
                "camera_intrinsics/extrinsics must be [B,T,V,3,3]/[B,T,V,4,4].")

        if intrinsics_hw is not None:
            h_cal, w_cal = float(intrinsics_hw[0]), float(intrinsics_hw[1])
        else:
            h_cal, w_cal = float(h), float(w)
        px = (xx / max(w, 1)) * w_cal
        py = (yy / max(h, 1)) * h_cal
        fx = k[..., 0, 0].clamp(min=1e-6).view(b, t, v, 1, 1)
        fy = k[..., 1, 1].clamp(min=1e-6).view(b, t, v, 1, 1)
        cx = k[..., 0, 2].view(b, t, v, 1, 1)
        cy = k[..., 1, 2].view(b, t, v, 1, 1)
        x_cam = (px.view(1, 1, 1, h, w) - cx) / fx
        y_cam = (py.view(1, 1, 1, h, w) - cy) / fy
        ones = torch.ones_like(x_cam)
        ray_cam = torch.stack((x_cam, y_cam, ones), dim=-1)
        ray_cam = F.normalize(ray_cam, dim=-1)

        rot = e[..., :3, :3]
        ray_ego = torch.einsum("btvij,btvhwj->btvhwi", rot, ray_cam)
        ray_ego = F.normalize(ray_ego, dim=-1)
        center = e[..., :3, 3].view(b, t, v, 1, 1, 3).expand(
            b, t, v, h, w, 3)
        return torch.cat([uv_norm, ray_ego, center], dim=-1)

    def forward(
        self,
        shape: tuple[int, int, int, int, int],
        dtype,
        device,
        intrinsics: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
        intrinsics_hw: Optional[tuple[int, int]],
    ) -> torch.Tensor:
        b, t, v, h, w = shape
        geom = self._camera_features(
            b, t, v, h, w, dtype, device,
            intrinsics, extrinsics, intrinsics_hw)
        return self.mlp(geom).permute(0, 1, 2, 5, 3, 4).contiguous()


class SceneAggregator(nn.Module):
    """Aggregate variable camera observations into fixed scene tokens."""

    def __init__(self, channels: int, heads: int, scene_tokens: int):
        super().__init__()
        self.scene_tokens = int(scene_tokens)
        self.query = nn.Parameter(torch.randn(self.scene_tokens, channels) * 0.02)
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B,T,V,C,H,W] -> [B,T,S,C,H,W].
        b, t, v, c, h, w = obs.shape
        kv = obs.permute(0, 1, 4, 5, 2, 3).reshape(b * t * h * w, v, c)
        q = self.query.view(1, self.scene_tokens, c).expand(
            b * t * h * w, -1, -1)
        kv_norm = self.norm_kv(kv)
        attn, _ = self.attn(self.norm_q(q), kv_norm, kv_norm)
        out = q + attn
        out = out + self.ff(out)
        return out.reshape(b, t, h, w, self.scene_tokens, c).permute(
            0, 1, 4, 5, 2, 3).contiguous()


class CameraRenderer(nn.Module):
    """Read fixed scene tokens into target camera feature tokens."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, scene: torch.Tensor, camera_query: torch.Tensor) -> torch.Tensor:
        # scene: [B,T,S,C,H,W], camera_query: [B,T,V,C,H,W].
        b, t, s, c, h, w = scene.shape
        v = camera_query.shape[2]
        kv = scene.permute(0, 1, 4, 5, 2, 3).reshape(b * t * h * w, s, c)
        q = camera_query.permute(0, 1, 4, 5, 2, 3).reshape(
            b * t * h * w, v, c)
        kv_norm = self.norm_kv(kv)
        attn, _ = self.attn(self.norm_q(q), kv_norm, kv_norm)
        out = q + attn
        out = out + self.ff(out)
        return out.reshape(b, t, h, w, v, c).permute(
            0, 1, 4, 5, 2, 3).contiguous()


class JointTVCompressor(nn.Module):
    def __init__(self, channels: int, heads: int, latent_t: int, latent_s: int):
        super().__init__()
        self.latent_t = int(latent_t)
        self.latent_s = int(latent_s)
        self.time_query = nn.Parameter(torch.randn(self.latent_t, channels) * 0.02)
        self.scene_query = nn.Parameter(torch.randn(self.latent_s, channels) * 0.02)
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,S,C,H,W] -> [B,T_lat,S_lat,C,H,W].
        b, t, s, c, h, w = x.shape
        kv = x.permute(0, 4, 5, 1, 2, 3).reshape(b * h * w, t * s, c)
        mean = kv.mean(dim=1, keepdim=True)
        q_seed = (
            self.time_query[:, None, :] + self.scene_query[None, :, :]
        ).reshape(1, self.latent_t * self.latent_s, c)
        q = mean + q_seed
        kv_norm = self.norm_kv(kv)
        attn, _ = self.attn(self.norm_q(q), kv_norm, kv_norm)
        out = q + attn
        out = out + self.ff(out)
        return out.reshape(b, h, w, self.latent_t, self.latent_s, c).permute(
            0, 3, 4, 5, 1, 2).contiguous()


class JointTVExpander(nn.Module):
    def __init__(self, channels: int, heads: int, latent_t: int, latent_s: int,
                 target_t: int, target_s: int):
        super().__init__()
        self.latent_t = int(latent_t)
        self.latent_s = int(latent_s)
        self.target_t = int(target_t)
        self.target_s = int(target_s)
        self.time_query = nn.Parameter(torch.randn(self.target_t, channels) * 0.02)
        self.scene_query = nn.Parameter(torch.randn(self.target_s, channels) * 0.02)
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,T_lat,S_lat,C,H,W] -> [B,T,S,C,H,W].
        b, t_lat, s_lat, c, h, w = z.shape
        if (t_lat, s_lat) != (self.latent_t, self.latent_s):
            raise ValueError(
                f"Expected latent {(self.latent_t, self.latent_s)}, "
                f"got {(t_lat, s_lat)}.")
        kv = z.permute(0, 4, 5, 1, 2, 3).reshape(
            b * h * w, self.latent_t * self.latent_s, c)
        mean = kv.mean(dim=1, keepdim=True)
        q_seed = (
            self.time_query[:, None, :] + self.scene_query[None, :, :]
        ).reshape(1, self.target_t * self.target_s, c)
        q = mean + q_seed
        kv_norm = self.norm_kv(kv)
        attn, _ = self.attn(self.norm_q(q), kv_norm, kv_norm)
        out = q + attn
        out = out + self.ff(out)
        return out.reshape(b, h, w, self.target_t, self.target_s, c).permute(
            0, 3, 4, 5, 1, 2).contiguous()


class NonJointTVCompressor(nn.Module):
    """Separable T/scene compression baseline, no T-scene attention."""

    def __init__(self, channels: int, latent_t: int, latent_s: int):
        super().__init__()
        self.latent_t = int(latent_t)
        self.latent_s = int(latent_s)
        self.refine = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, s, c, h, w = x.shape
        y = x.permute(0, 4, 5, 3, 1, 2).reshape(b * h * w, c, t, s)
        y = F.adaptive_avg_pool2d(y, (self.latent_t, self.latent_s))
        y = y.permute(0, 2, 3, 1).reshape(
            b, h, w, self.latent_t, self.latent_s, c)
        y = y + self.refine(y)
        return y.permute(0, 3, 4, 5, 1, 2).contiguous()


class NonJointTVExpander(nn.Module):
    def __init__(self, channels: int, target_t: int, target_s: int):
        super().__init__()
        self.target_t = int(target_t)
        self.target_s = int(target_s)
        self.refine = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, _t, _s, c, h, w = z.shape
        y = z.permute(0, 4, 5, 3, 1, 2).reshape(
            b * h * w, c, z.shape[1], z.shape[2])
        y = F.interpolate(
            y, size=(self.target_t, self.target_s),
            mode="bilinear", align_corners=False)
        y = y.permute(0, 2, 3, 1).reshape(
            b, h, w, self.target_t, self.target_s, c)
        y = y + self.refine(y)
        return y.permute(0, 3, 4, 5, 1, 2).contiguous()


def _spatial_levels(factor: int) -> int:
    factor = int(factor)
    if factor < 1 or factor & (factor - 1):
        raise ValueError(
            "spatial_downsample_factor must be a positive power of two, "
            f"got {factor}.")
    return int(math.log2(factor))


class TVVAE(nn.Module):
    """Camera-conditioned scene-token VAE for TV compression experiments."""

    def __init__(self, **config):
        super().__init__()
        self.config = TVVAEConfig(**config)
        cfg = self.config
        if cfg.tv_compression not in ("joint", "non_joint"):
            raise ValueError(
                "tv_compression must be 'joint' or 'non_joint', got "
                f"{cfg.tv_compression}.")
        if cfg.latent_time_count > cfg.sequence_length:
            raise ValueError("latent_time_count must be <= sequence_length.")
        if cfg.latent_scene_token_count > cfg.scene_token_count:
            raise ValueError(
                "latent_scene_token_count must be <= scene_token_count.")

        levels = _spatial_levels(cfg.spatial_downsample_factor)
        self.stem = CausalConv3d(cfg.in_channels, cfg.base_channels, 3, padding=1)
        channels = cfg.base_channels
        down_blocks = []
        enc_channels = [channels]
        for level in range(levels):
            out_channels = cfg.base_channels * min(2 ** (level + 1), 4)
            down_blocks.append(ResDownBlock(channels, out_channels))
            channels = out_channels
            enc_channels.append(channels)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bottleneck_channels = channels

        heads = int(cfg.num_attention_heads)
        if channels % heads != 0:
            raise ValueError(
                f"bottleneck channels {channels} must be divisible by heads {heads}.")

        self.geometry = GeometryEmbedding(channels)
        self.scene_aggregator = SceneAggregator(
            channels, heads, cfg.scene_token_count)
        self.camera_renderer = CameraRenderer(channels, heads)
        self.target_camera_fallback = nn.Parameter(
            torch.randn(cfg.view_count, channels) * 0.02)

        if cfg.tv_compression == "joint":
            self.compress = JointTVCompressor(
                channels, heads,
                cfg.latent_time_count,
                cfg.latent_scene_token_count)
            self.expand = JointTVExpander(
                channels, heads,
                cfg.latent_time_count,
                cfg.latent_scene_token_count,
                cfg.sequence_length,
                cfg.scene_token_count)
        else:
            self.compress = NonJointTVCompressor(
                channels,
                cfg.latent_time_count,
                cfg.latent_scene_token_count)
            self.expand = NonJointTVExpander(
                channels,
                cfg.sequence_length,
                cfg.scene_token_count)

        self.to_moments = nn.Linear(channels, cfg.latent_channels * 2)
        self.from_latent = nn.Linear(cfg.latent_channels, channels)

        up_blocks = []
        for out_channels in reversed(enc_channels[:-1]):
            up_blocks.append(ResUpBlock(channels, out_channels))
            channels = out_channels
        self.up_blocks = nn.ModuleList(up_blocks)
        self.head = CausalConv3d(channels, cfg.out_channels, 3, padding=1)
        self._last_hw: Optional[tuple[int, int]] = None
        self._last_view_count = cfg.view_count
        self._last_intrinsics: Optional[torch.Tensor] = None
        self._last_extrinsics: Optional[torch.Tensor] = None
        self._last_intrinsics_hw: Optional[tuple[int, int]] = None

    def _flatten_views(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1, 4, 5).reshape(
            x.shape[0] * x.shape[2], x.shape[3], x.shape[1],
            x.shape[4], x.shape[5])

    def _unflatten_views(self, y: torch.Tensor, b: int, v: int) -> torch.Tensor:
        return y.reshape(
            b, v, y.shape[1], y.shape[2], y.shape[3], y.shape[4]
        ).permute(0, 3, 1, 2, 4, 5).contiguous()

    def _camera_query(
        self,
        b: int,
        t: int,
        v: int,
        h: int,
        w: int,
        dtype,
        device,
        intrinsics: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
        intrinsics_hw: Optional[tuple[int, int]],
    ) -> torch.Tensor:
        geom = self.geometry(
            (b, t, v, h, w), dtype, device,
            intrinsics, extrinsics, intrinsics_hw)
        if v <= self.target_camera_fallback.shape[0]:
            learned = self.target_camera_fallback[:v].to(
                device=device, dtype=dtype).view(1, 1, v, -1, 1, 1)
            geom = geom + learned
        return geom

    def encode(
        self,
        x: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics_hw: Optional[tuple[int, int]] = None,
    ) -> DiagonalGaussianDistribution:
        if x.ndim != 6:
            raise ValueError(f"TVVAE expects [B,T,V,C,H,W], got {tuple(x.shape)}")
        b, t, v, _c, h_in, w_in = x.shape
        y = self._flatten_views(x)
        y = self.stem(y)
        for block in self.down_blocks:
            y = block(y)
        obs = self._unflatten_views(y, b, v)
        self._last_hw = (int(obs.shape[-2]), int(obs.shape[-1]))
        self._last_view_count = v
        self._last_intrinsics = intrinsics
        self._last_extrinsics = extrinsics
        self._last_intrinsics_hw = intrinsics_hw
        self.config.image_height = h_in
        self.config.image_width = w_in

        geom = self.geometry(
            (b, t, v, obs.shape[-2], obs.shape[-1]),
            obs.dtype, obs.device,
            intrinsics, extrinsics, intrinsics_hw)
        scene = self.scene_aggregator(obs + geom)
        latent_feat = self.compress(scene)
        moments = self.to_moments(latent_feat.permute(0, 1, 2, 4, 5, 3))
        mean, logvar = moments.chunk(2, dim=-1)
        mean = mean.permute(0, 1, 2, 5, 3, 4).contiguous()
        logvar = logvar.permute(0, 1, 2, 5, 3, 4).contiguous()
        return DiagonalGaussianDistribution(mean, logvar)

    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics_hw: Optional[tuple[int, int]] = None,
        target_view_count: Optional[int] = None,
    ):
        if z.ndim != 6:
            raise ValueError(
                "TVVAE latent must be [B,T_lat,S_lat,C,H,W], got "
                f"{tuple(z.shape)}.")
        hw = self._last_hw
        if hw is None:
            if self.config.image_height is None or self.config.image_width is None:
                raise ValueError("Cannot infer bottleneck H/W before encode.")
            factor = int(self.config.spatial_downsample_factor)
            hw = (
                math.ceil(int(self.config.image_height) / factor),
                math.ceil(int(self.config.image_width) / factor),
            )
        if tuple(z.shape[-2:]) != hw:
            raise ValueError(
                f"Latent spatial shape {tuple(z.shape[-2:])} != expected {hw}.")

        h = self.from_latent(z.permute(0, 1, 2, 4, 5, 3))
        h = h.permute(0, 1, 2, 5, 3, 4).contiguous()
        scene = self.expand(h)
        b, t, _s, c, h_lat, w_lat = scene.shape
        v = int(target_view_count or self._last_view_count)
        intrinsics = self._last_intrinsics if intrinsics is None else intrinsics
        extrinsics = self._last_extrinsics if extrinsics is None else extrinsics
        intrinsics_hw = (
            self._last_intrinsics_hw if intrinsics_hw is None else intrinsics_hw)
        cam_query = self._camera_query(
            b, t, v, h_lat, w_lat, scene.dtype, scene.device,
            intrinsics, extrinsics, intrinsics_hw)
        cam_feat = self.camera_renderer(scene, cam_query)
        y = self._flatten_views(cam_feat)
        for block in self.up_blocks:
            y = block(y)
        y = self.head(y)
        out = self._unflatten_views(y, b, v)
        if (
            self.config.image_height is not None and
            self.config.image_width is not None and
            out.shape[-2:] != (self.config.image_height, self.config.image_width)
        ):
            flat = out.reshape(
                out.shape[0] * out.shape[1] * out.shape[2],
                out.shape[3], out.shape[4], out.shape[5])
            flat = F.interpolate(
                flat,
                size=(int(self.config.image_height), int(self.config.image_width)),
                mode="bilinear",
                align_corners=False)
            out = flat.reshape(
                out.shape[0], out.shape[1], out.shape[2],
                self.config.out_channels,
                int(self.config.image_height), int(self.config.image_width))
        if return_dict:
            return DecoderOutput(out)
        return (out,)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = False,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics_hw: Optional[tuple[int, int]] = None,
        **_unused,
    ) -> dict:
        posterior = self.encode(x, intrinsics, extrinsics, intrinsics_hw)
        z = posterior.sample() if sample_posterior else posterior.mode()
        recon = self.decode(
            z,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            intrinsics_hw=intrinsics_hw,
            target_view_count=x.shape[2],
        ).sample
        return {
            "posterior": posterior,
            "latent_sample": z,
            "sample": recon,
        }
