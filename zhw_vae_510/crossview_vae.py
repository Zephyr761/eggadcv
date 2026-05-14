import json
import math
import os
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.logvar = logvar.clamp(-30.0, 20.0)

    def sample(self):
        return self.mean + torch.randn_like(self.mean) * torch.exp(
            0.5 * self.logvar)

    def mode(self):
        return self.mean

    def kl(self):
        return -0.5 * torch.sum(
            1 + self.logvar - self.mean.pow(2) - self.logvar.exp())


class EncoderOutput:
    def __init__(self, latent_dist: DiagonalGaussianDistribution):
        self.latent_dist = latent_dist


class DecoderOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample

    def __getitem__(self, index):
        if index == 0:
            return self.sample
        raise IndexError(index)


def _namespace_config(config: Optional[dict]):
    config = {} if config is None else dict(config)
    config.setdefault("vae_type", "crossview_4d")
    config.setdefault("in_channels", 3)
    config.setdefault("out_channels", 3)
    config.setdefault("base_channels", 64)
    config.setdefault("latent_channels", 16)
    config.setdefault("virtual_view_count", 4)
    config.setdefault("latent_view_count", 3)
    config.setdefault("temporal_downsample_factor", 4)
    config.setdefault("temporal_pre", 1)
    config.setdefault("spatial_downsample_factor", 8)
    config.setdefault("num_bottleneck_blocks", 2)
    config.setdefault("num_attention_heads", 2)
    config.setdefault("scaling_factor", 0.18215)
    config.setdefault("shift_factor", 0.0)

    spatial_levels = int(math.log2(config["spatial_downsample_factor"]))
    config["block_out_channels"] = tuple(
        config.get(
            "block_out_channels",
            [config["base_channels"] * min(2 ** i, 4)
             for i in range(spatial_levels + 1)]))
    config["down_block_types"] = tuple(
        config.get("down_block_types", ["CrossViewDownBlock"] *
                   (spatial_levels + 1)))
    return SimpleNamespace(**config)


class CylindricalViewProjector(nn.Module):
    """Project input camera features onto a fixed virtual cylindrical rig.

    Two paths are supported:
    1. Geometry-aware path (preferred): when ``intrinsics`` and ``extrinsics``
       are provided, the module casts rays from a unit cylinder around the ego
       car, projects them into every input camera, and aggregates samples with
       a soft FOV visibility mask via ``grid_sample``. This is a real
       multi-view geometry operation (no ghosting from pixel-level mixing of
       unrelated cameras).
    2. Fallback path: when no camera parameters are provided we assume the
       input cameras tile a 360 deg cylinder uniformly and resample each
       virtual view from a single nearest input camera. We deliberately do
       **not** linearly blend pixels of two different physical cameras here,
       because that is exactly the ghosting failure mode we want to avoid.

    The projection happens at the feature level (i.e. after the encoder stem
    or any cheap convolution that maps RGB to a feature map). Doing it on a
    feature grid keeps the cost manageable while still being a true geometric
    re-binning rather than a 2D image blend.
    """

    def __init__(
        self,
        virtual_view_count: int,
        fov_deg: float = 110.0,
        cylinder_radii=(10.0,),
    ):
        super().__init__()
        self.virtual_view_count = virtual_view_count
        # FOV used both for fallback panorama tiling and for the soft
        # visibility mask in the ray-projection path.
        self.fov = math.radians(fov_deg)
        # Reference cylinder radii (in metres, matching the unit of the
        # extrinsics translation column). Using a *finite* radius is critical
        # for autonomous-driving footage: with the previous "rays through the
        # origin" assumption, the camera translation was ignored entirely,
        # which is equivalent to assuming infinite depth and produces severe
        # ghosting for nearby objects (e.g. cars 1-5 m away). With a finite
        # radius R, an object at depth R projects exactly; objects at other
        # depths still suffer parallax error, but the magnitude is ``1/R``
        # times smaller than at ``R = inf``.
        #
        # Multiple radii can be supplied to mitigate the near/far trade-off
        # via simple-bev / LSS-style multi-depth aggregation: rays are cast
        # at each depth, sampled features are averaged with their visibility
        # weights, and the network can pick whichever depth aligns best for
        # each spatial location.
        if isinstance(cylinder_radii, (int, float)):
            cylinder_radii = (float(cylinder_radii),)
        self.register_buffer(
            "cylinder_radii",
            torch.tensor(list(cylinder_radii), dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _virtual_azimuth(virtual_view_count: int, device, dtype):
        # Returns the azimuth (rad) of the optical axis of each virtual view,
        # uniformly distributed around 360 deg.
        return (torch.arange(virtual_view_count, device=device, dtype=dtype)
                + 0.5) * (2 * math.pi / virtual_view_count) - math.pi

    def _ray_project(
        self,
        x: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics_hw: Optional["tuple[int, int]"] = None,
    ) -> torch.Tensor:
        # x: [B, T, V_in, C, H, W]
        # intrinsics: [B, T, V_in, 3, 3] (or broadcastable)
        # extrinsics (camera_to_ego): [B, T, V_in, 4, 4]
        # intrinsics_hw: original (H_calib, W_calib) the intrinsics were
        #   calibrated against. We need it because intrinsics live in the
        #   pixel coordinate frame of the *original* image (e.g. 1920x1080),
        #   while ``x`` here is a feature map at a possibly very different
        #   resolution (e.g. 256x256, or even 32x32 after stem+downsamples).
        #   Without this rescale, ``u_img`` stays at the calibration scale
        #   while ``w`` is the feature scale, so ``gx, gy`` blow past
        #   ``[-1, 1]`` and ``grid_sample`` returns all-zeros.
        b, t, v_in, c, h, w = x.shape
        v_out = self.virtual_view_count
        device = x.device
        dtype = x.dtype

        # Build a cylindrical sampling grid in the ego frame for each
        # virtual view: a (H, W) patch of azimuth phi and height y around
        # the virtual camera's optical axis.
        center_phi = self._virtual_azimuth(v_out, device, dtype)  # [V_out]
        # Each virtual view covers exactly its share of the cylinder so that
        # adjacent virtual views form a continuous panorama.
        delta_phi = 2 * math.pi / v_out
        u = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w - 0.5
        v_lin = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h - 0.5
        # phi: [V_out, H, W], y: [V_out, H, W]
        phi = center_phi.view(v_out, 1, 1) + u.view(1, 1, w) * delta_phi
        # Vertical extent on the cylinder is set so the aspect ratio matches
        # the angular extent times h/w.
        y_extent = delta_phi * (h / max(w, 1))
        y_grid = (-v_lin).view(1, h, 1) * y_extent
        y_grid = y_grid.expand(v_out, h, w)
        # Ray directions in ego frame (cylinder of radius 1).
        phi = phi.expand(v_out, h, w)
        rx = torch.sin(phi)
        rz = torch.cos(phi)
        ry = y_grid
        rays = torch.stack([rx, ry, rz], dim=-1)  # [V_out, H, W, 3]

        # Broadcast intrinsics/extrinsics to [B, T, V_in, ...].
        if intrinsics.dim() == 4:
            intrinsics = intrinsics.unsqueeze(1).expand(b, t, v_in, 3, 3)
        if extrinsics.dim() == 4:
            extrinsics = extrinsics.unsqueeze(1).expand(b, t, v_in, 4, 4)

        # Full ego -> camera transform (NOT just rotation): we now place
        # cylinder samples at a finite depth so the camera translation
        # actually matters. Using only the rotation, as in the previous
        # version, is equivalent to ``cylinder_radius = inf`` and produces
        # severe ghosting for objects that are close to the rig.
        ext_inv = torch.linalg.inv(extrinsics.float()).to(dtype)
        # Translation column of camera_to_ego = camera position in ego frame.
        cam_pos_ego = extrinsics[..., :3, 3]                      # [B,T,V_in,3]
        # Per-camera principal axis in ego frame (camera +Z direction).
        cam_z_ego = extrinsics[..., :3, 2]                        # [B,T,V_in,3]

        # Pre-compute per-camera intrinsic scales (calibration -> feature map).
        if intrinsics_hw is not None:
            h_cal = float(intrinsics_hw[0])
            w_cal = float(intrinsics_hw[1])
            sx = w / max(w_cal, 1.0)
            sy = h / max(h_cal, 1.0)
            fx = intrinsics[..., 0, 0:1] * sx
            fy = intrinsics[..., 1, 1:2] * sy
            cx = intrinsics[..., 0, 2:3] * sx
            cy = intrinsics[..., 1, 2:3] * sy
        else:
            cx_raw = intrinsics[..., 0, 2:3]
            cy_raw = intrinsics[..., 1, 2:3]
            sx = w / (2.0 * cx_raw.clamp(min=1e-3))
            sy = h / (2.0 * cy_raw.clamp(min=1e-3))
            fx = intrinsics[..., 0, 0:1] * sx
            fy = intrinsics[..., 1, 1:2] * sy
            cx = cx_raw * sx
            cy = cy_raw * sy

        rays_flat = rays.reshape(-1, 3)                           # [N, 3]
        feat = x.reshape(b * t * v_in, c, h, w)

        # Multi-depth aggregation: cast rays at each ``cylinder_radius``,
        # weighted-average the resulting samples. With a single radius this
        # collapses to a plain finite-depth projection. Using multiple radii
        # lets the network pick whichever depth aligns best at each spatial
        # location.
        out_acc = None
        weight_acc = None
        for r in self.cylinder_radii.tolist():
            # 3D points on the cylinder of radius r in the ego frame.
            points_ego = rays_flat * r                           # [N, 3]
            ones = torch.ones_like(points_ego[..., :1])
            points_h = torch.cat([points_ego, ones], dim=-1)     # [N, 4]
            # Transform to each camera frame: [B,T,V_in,N,4]
            points_cam_h = torch.einsum(
                "btvij,nj->btvni", ext_inv, points_h)
            points_cam = points_cam_h[..., :3]                   # [B,T,V_in,N,3]

            z_cam = points_cam[..., 2]
            valid_front = z_cam > 1e-3
            x_pix = points_cam[..., 0] / z_cam.clamp(min=1e-3)
            y_pix = points_cam[..., 1] / z_cam.clamp(min=1e-3)
            u_img = x_pix * fx + cx                              # [B,T,V_in,N]
            v_img = y_pix * fy + cy
            gx = (u_img / max(w, 1)) * 2 - 1
            gy = (v_img / max(h, 1)) * 2 - 1
            in_image = (gx.abs() < 1) & (gy.abs() < 1) & valid_front

            # Visibility weight: angle between (camera -> point) vector and
            # the camera's principal axis in the ego frame, masked by FOV.
            # Using the proper from-camera vector (rather than the from-ego
            # ray direction) gives a more accurate FOV mask when the camera
            # is offset from the ego origin.
            vec = points_ego.view(1, 1, 1, -1, 3) \
                - cam_pos_ego.unsqueeze(-2)                      # [B,T,V_in,N,3]
            vec_n = vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            cos_axis = (vec_n * cam_z_ego.unsqueeze(-2)).sum(-1)  # [B,T,V_in,N]
            weight = (cos_axis - math.cos(self.fov / 2)).clamp(min=0)
            weight = weight * in_image.to(weight.dtype)

            # Sample features from each input camera at this depth.
            grid = torch.stack([gx, gy], dim=-1).reshape(
                b * t * v_in, v_out, h * w, 2)
            sampled = F.grid_sample(
                feat, grid, mode="bilinear", padding_mode="zeros",
                align_corners=False)
            sampled = sampled.view(b, t, v_in, c, v_out, h, w)
            weight = weight.view(b, t, v_in, 1, v_out, h, w)

            if out_acc is None:
                out_acc = sampled * weight
                weight_acc = weight
            else:
                out_acc = out_acc + sampled * weight
                weight_acc = weight_acc + weight

        # Aggregate over input cameras and depths.
        weight_sum = weight_acc.sum(dim=2).clamp(min=1e-4)
        out = out_acc.sum(dim=2) / weight_sum
        # out is [B, T, C, V_out, H, W] -> permute to [B, T, V_out, C, H, W].
        return out.permute(0, 1, 3, 2, 4, 5).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics_hw: Optional["tuple[int, int]"] = None,
    ) -> torch.Tensor:
        # x: [B, T, V_in, C, H, W]
        if intrinsics is not None and extrinsics is not None:
            return self._ray_project(x, intrinsics, extrinsics, intrinsics_hw)

        v_in = x.shape[2]
        v_out = self.virtual_view_count
        if v_in == v_out:
            return x
        # Fallback: nearest-camera assignment (no inter-camera pixel blend).
        idx = (torch.arange(v_out, device=x.device, dtype=torch.float32)
               + 0.5) * v_in / v_out
        idx = idx.floor().long().clamp(0, v_in - 1)
        return x[:, :, idx]


class CircularConv3d(nn.Module):
    """3D conv whose first spatial axis is the circular (panoramic) view axis.

    Input layout: ``[N, C, V, H, W]``. The view axis is wrapped with circular
    padding so the convolution treats the multi-view rig as a closed ring,
    while H/W use standard zero padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=0, bias=bias)

    def forward(self, x: torch.Tensor):
        # x: [N, C, V, H, W]
        pv, ph, pw = [i // 2 for i in self.kernel_size]
        if ph or pw:
            x = F.pad(x, (pw, pw, ph, ph, 0, 0))
        if pv:
            x = F.pad(x, (0, 0, 0, 0, pv, pv), mode="circular")
        return self.conv(x)


class CausalConv3d(nn.Conv3d):
    """3D convolution with causal padding on the time axis (Wan2.1 style).

    Input layout: ``[N, C, T, H, W]``. The full temporal padding budget
    ``2 * (kernel_t // 2)`` is pushed to the **left** so that ``output[t]``
    only depends on ``input[<= t]`` (after stride). H/W use standard
    symmetric padding inherited from ``nn.Conv3d``.

    Like Wan2.1's CausalConv3d, the layer accepts an optional ``cache_x``
    argument: at autoregressive / streaming inference time the caller passes
    the last few frames from the previous chunk so this chunk's output is
    bit-identical to the non-streamed version. This is what makes the VAE
    safe to pair with a diffusion model that generates frames step by step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # nn.Conv3d.padding is (pT, pH, pW). Move the full temporal budget
        # to the left and zero out the temporal padding inside Conv3d itself.
        pT, pH, pW = self.padding
        self._causal_padding = (pW, pW, pH, pH, 2 * pT, 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None):
        padding = list(self._causal_padding)
        if cache_x is not None and padding[4] > 0:
            cache_x = cache_x.to(x.device, dtype=x.dtype)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] = max(padding[4] - cache_x.shape[2], 0)
        x = F.pad(x, padding)
        return super().forward(x)


def _build_view_coords(view_count: int, h: int, w: int, device, dtype,
                       ref_view_count: Optional[int] = None):
    """Compute global ego-cylinder coordinates for tokens of a multi-view grid.

    Returns ``(azimuth, height)`` tensors of shape ``[V*H*W]`` in
    **pseudo-pixel** units. Both axes are scaled so adjacent-token steps are
    O(1), which matters for RoPE: with ``freq = base^(-i/pairs)`` the highest
    band has frequency 1, so a coord step of e.g. 0.02 rad would only rotate
    that channel by 0.02 between neighbouring tokens, leaving the channel
    almost constant and useless for distinguishing nearby positions.

    - Height: pixel index ``0.5, 1.5, ..., H-0.5``. Step = 1 pixel.
    - Azimuth: physical angle scaled by ``ref_view_count * w / (2*pi)``, so
      one full revolution corresponds to ``ref_view_count * w`` pseudo-pixels
      and adjacent tokens within a view differ by exactly 1 pseudo-pixel.

    ``ref_view_count`` controls the azimuth scale and **must be shared
    between Q and K** in any cross-attention so tokens that look at the same
    physical azimuth (regardless of how many views they were tiled into)
    still receive the same RoPE phase. For self-attention pass
    ``ref_view_count = view_count`` (default).
    """
    if ref_view_count is None:
        ref_view_count = view_count
    # Build the azimuth coord directly in pseudo-pixel units, monotonically
    # increasing across views. Using a *monotonic* coord (rather than wrapped
    # into [-pi, pi]) is important: physically adjacent tokens that straddle
    # a view boundary then differ by exactly 1 pseudo-pixel, just like any
    # two adjacent tokens within a view. With wrapped coords, that boundary
    # would jump by ~ -V*w which RoPE would interpret as a huge distance.
    #
    # Each input view occupies ``ref_view_count * w / view_count`` pseudo-
    # pixels along the azimuth axis (so the full revolution is always
    # ``ref_view_count * w`` regardless of how the views are tiled).
    px_per_view = ref_view_count * w / float(view_count)
    v_idx = torch.arange(view_count, device=device, dtype=dtype)
    u_idx = torch.arange(w, device=device, dtype=dtype) + 0.5
    az = v_idx.view(view_count, 1, 1) * px_per_view \
        + u_idx.view(1, 1, w) * (px_per_view / w)
    az = az.expand(view_count, h, w).reshape(-1)
    y = (torch.arange(h, device=device, dtype=dtype) + 0.5)
    y = y.view(1, h, 1).expand(view_count, h, w).reshape(-1)
    return az, y


def _per_frame_groupnorm(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    """Apply a 2D :class:`nn.GroupNorm` per timestep on a 5D tensor.

    Input layout: ``[N, C, T, H, W]``. We fold ``T`` into the batch dimension
    before normalising so the running statistics are computed over
    ``(C/G, H, W)`` only. This is what makes the surrounding causal stack
    actually causal: a vanilla 3D ``GroupNorm`` would average over time and
    leak future frames into earlier outputs, defeating the point of
    :class:`CausalConv3d`.
    """
    n, c, t, h, w = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    y = norm(y)
    return y.view(n, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class TimeSpaceDownBlock(nn.Module):
    """Per-view residual block that downsamples ``(T, H, W)`` causally.

    Time uses :class:`CausalConv3d` so output frames only see past inputs.
    GroupNorm is applied per-frame (see :func:`_per_frame_groupnorm`) so the
    normalisation does not leak future frames into past outputs.
    """

    def __init__(self, channels: int, out_channels: int, stride_t: int,
                 stride_hw: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = CausalConv3d(
            channels, out_channels, 3,
            stride=(stride_t, stride_hw, stride_hw), padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        self.skip = CausalConv3d(
            channels, out_channels, 1,
            stride=(stride_t, stride_hw, stride_hw), padding=0)

    def forward(self, x: torch.Tensor):
        # x: [B, T, V, C, H, W]
        b, t, v, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1, 4, 5).reshape(b * v, c, t, h, w)
        skip = self.skip(y)
        h_y = self.conv1(F.silu(_per_frame_groupnorm(self.norm1, y)))
        h_y = self.conv2(F.silu(_per_frame_groupnorm(self.norm2, h_y)))
        y = h_y + skip
        _, c2, t2, h2, w2 = y.shape
        return y.reshape(b, v, c2, t2, h2, w2).permute(0, 3, 1, 2, 4, 5)


class TimeSpaceUpBlock(nn.Module):
    """Per-view residual block that upsamples (T, H, W) causally.

    The temporal upsample is implemented as nearest-neighbour repeat (also
    causal) followed by causal 3D convolutions, mirroring Wan2.1's decoder.
    """

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = CausalConv3d(channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, scale_t: int, scale_hw: int):
        b, t, v, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1, 4, 5).reshape(b * v, c, t, h, w)
        # Nearest-neighbour repeat in time keeps the upsample causal; for the
        # spatial axes we use trilinear (only along H/W, scale_t kept at 1).
        if scale_t != 1:
            y = y.repeat_interleave(scale_t, dim=2)
        if scale_hw != 1:
            y = F.interpolate(
                y, scale_factor=(1, scale_hw, scale_hw),
                mode="trilinear", align_corners=False)
        y = self.conv1(F.silu(_per_frame_groupnorm(self.norm1, y)))
        y = self.conv2(F.silu(_per_frame_groupnorm(self.norm2, y)))
        _, c2, t2, h2, w2 = y.shape
        return y.reshape(b, v, c2, t2, h2, w2).permute(0, 3, 1, 2, 4, 5)


class ViewMixResBlock(nn.Module):
    """Per-frame residual block mixing information across the circular view
    axis with :class:`CircularConv3d`.

    Operates per timestep so it composes cleanly with the causal time blocks:
    causality along T is preserved because no temporal mixing happens here.
    The convolution kernel covers ``(V, H, W)`` with circular padding on V,
    so neighbouring views on the cylinder share information physically (the
    front-left camera blends with the front and left cameras through wrap).
    """

    def __init__(self, channels: int, view_kernel: int = 3,
                 spatial_kernel: int = 3):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = CircularConv3d(
            channels, channels,
            kernel_size=(view_kernel, spatial_kernel, spatial_kernel))
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = CircularConv3d(
            channels, channels,
            kernel_size=(view_kernel, spatial_kernel, spatial_kernel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, V, C, H, W]
        b, t, v, c, h, w = x.shape
        y = x.permute(0, 1, 3, 2, 4, 5).reshape(b * t, c, v, h, w)
        h_y = self.conv1(F.silu(self.norm1(y)))
        h_y = self.conv2(F.silu(self.norm2(h_y)))
        y = y + h_y
        return y.reshape(b, t, c, v, h, w).permute(0, 1, 3, 2, 4, 5)\
            .contiguous()


class ViewQueryCompressor(nn.Module):
    """Compress ``V_in`` virtual views into ``V_out`` latent views via
    cross-attention with learnable view-query tokens.

    Each output (latent view, spatial position) token is a learnable query
    that cross-attends to all input (virtual view, spatial position) tokens.
    Both queries and keys carry global cylinder RoPE on (azimuth, height), so
    a query at azimuth ``phi`` attends most strongly to keys at the same
    physical azimuth, regardless of which input view they came from. This is
    the mechanism that lets overlapping content from different virtual views
    collapse into a single latent token without the cliff-like artifacts of
    a strided convolution along the view axis.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        latent_views: int,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.latent_views = latent_views
        self.view_query_emb = nn.Parameter(
            torch.randn(latent_views, channels) * 0.02)
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(channels, channels * 2)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.rope_base = rope_base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, V_in, C, H, W]
        b, t, v_in, c, h, w = x.shape
        v_out = self.latent_views
        # Initialize latent queries from a smooth statistic of the inputs
        # (mean over views) plus a learnable per-latent-view embedding.
        mean = x.mean(dim=2, keepdim=True)
        latent = mean.expand(-1, -1, v_out, -1, -1, -1)
        view_emb = self.view_query_emb.view(1, 1, v_out, c, 1, 1)
        latent = latent + view_emb

        q_tokens = latent.permute(0, 1, 2, 4, 5, 3).reshape(
            b * t, v_out * h * w, c)
        kv_tokens = x.permute(0, 1, 2, 4, 5, 3).reshape(
            b * t, v_in * h * w, c)

        q = self.to_q(self.norm_q(q_tokens))
        kv = self.to_kv(self.norm_kv(kv_tokens))
        k, v_feat = kv.chunk(2, dim=-1)

        q = q.view(b * t, v_out * h * w, self.num_heads, self.head_dim)\
            .transpose(1, 2)
        k = k.view(b * t, v_in * h * w, self.num_heads, self.head_dim)\
            .transpose(1, 2)
        v_feat = v_feat.view(
            b * t, v_in * h * w, self.num_heads, self.head_dim).transpose(1, 2)

        # Shared azimuth scale: use the larger of (v_in, v_out) as reference
        # so tokens at the same physical azimuth get identical RoPE phases
        # on both Q and K side, regardless of how many views they were
        # tiled into.
        ref_v = max(v_in, v_out)
        q_az, q_h = _build_view_coords(
            v_out, h, w, x.device, q.dtype, ref_view_count=ref_v)
        k_az, k_h = _build_view_coords(
            v_in, h, w, x.device, k.dtype, ref_view_count=ref_v)

        pairs = max(self.head_dim // 4, 1) if self.head_dim >= 4 else 0
        offset = 0
        if pairs:
            q, k = _apply_axis_rope_qk(
                q, k, q_az, k_az, offset, pairs, base=self.rope_base)
            offset += pairs * 2
            if offset + pairs * 2 <= self.head_dim:
                q, k = _apply_axis_rope_qk(
                    q, k, q_h, k_h, offset, pairs, base=self.rope_base)

        attn = F.scaled_dot_product_attention(q, k, v_feat)
        attn = attn.transpose(1, 2).reshape(b * t, v_out * h * w, c)
        out = q_tokens + self.proj(attn)
        out = out.reshape(b, t, v_out, h, w, c).permute(0, 1, 2, 5, 3, 4)
        return out.contiguous()


class ViewQueryExpander(nn.Module):
    """Mirror of :class:`ViewQueryCompressor` for the decoder.

    Re-expands ``V_in`` latent views back to ``V_out`` virtual views via
    cross-attention with learnable per-virtual-view query tokens.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        virtual_views: int,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.virtual_views = virtual_views
        self.view_query_emb = nn.Parameter(
            torch.randn(virtual_views, channels) * 0.02)
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(channels, channels * 2)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.rope_base = rope_base

    def forward(self, x: torch.Tensor, target_views: Optional[int] = None
                ) -> torch.Tensor:
        b, t, v_in, c, h, w = x.shape
        v_out = target_views if target_views is not None else self.virtual_views
        if v_out > self.virtual_views:
            raise ValueError(
                f"ViewQueryExpander built for at most {self.virtual_views} "
                f"virtual views, got target_views={v_out}.")
        # Initial queries: broadcast latent mean + per-virtual-view embedding.
        mean = x.mean(dim=2, keepdim=True)
        queries = mean.expand(-1, -1, v_out, -1, -1, -1)
        view_emb = self.view_query_emb[:v_out].view(1, 1, v_out, c, 1, 1)
        queries = queries + view_emb

        q_tokens = queries.permute(0, 1, 2, 4, 5, 3).reshape(
            b * t, v_out * h * w, c)
        kv_tokens = x.permute(0, 1, 2, 4, 5, 3).reshape(
            b * t, v_in * h * w, c)

        q = self.to_q(self.norm_q(q_tokens))
        kv = self.to_kv(self.norm_kv(kv_tokens))
        k, v_feat = kv.chunk(2, dim=-1)

        q = q.view(b * t, v_out * h * w, self.num_heads, self.head_dim)\
            .transpose(1, 2)
        k = k.view(b * t, v_in * h * w, self.num_heads, self.head_dim)\
            .transpose(1, 2)
        v_feat = v_feat.view(
            b * t, v_in * h * w, self.num_heads, self.head_dim).transpose(1, 2)

        ref_v = max(v_in, v_out)
        q_az, q_h = _build_view_coords(
            v_out, h, w, x.device, q.dtype, ref_view_count=ref_v)
        k_az, k_h = _build_view_coords(
            v_in, h, w, x.device, k.dtype, ref_view_count=ref_v)

        pairs = max(self.head_dim // 4, 1) if self.head_dim >= 4 else 0
        offset = 0
        if pairs:
            q, k = _apply_axis_rope_qk(
                q, k, q_az, k_az, offset, pairs, base=self.rope_base)
            offset += pairs * 2
            if offset + pairs * 2 <= self.head_dim:
                q, k = _apply_axis_rope_qk(
                    q, k, q_h, k_h, offset, pairs, base=self.rope_base)

        attn = F.scaled_dot_product_attention(q, k, v_feat)
        attn = attn.transpose(1, 2).reshape(b * t, v_out * h * w, c)
        out = q_tokens + self.proj(attn)
        out = out.reshape(b, t, v_out, h, w, c).permute(0, 1, 2, 5, 3, 4)
        return out.contiguous()


def _rotate_half(x: torch.Tensor):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _rope_sin_cos(coords: torch.Tensor, pairs: int, base: float, dtype):
    device = coords.device
    freq = torch.arange(pairs, device=device, dtype=torch.float32)
    freq = base ** (-freq / max(pairs, 1))
    angle = coords.to(torch.float32).unsqueeze(-1) * freq
    width = pairs * 2
    sin = angle.sin().repeat_interleave(2, dim=-1).view(1, 1, -1, width)
    cos = angle.cos().repeat_interleave(2, dim=-1).view(1, 1, -1, width)
    return sin.to(dtype), cos.to(dtype)


def _apply_axis_rope(q, k, coords, offset, pairs, base=10000.0):
    """In-place RoPE for matched-length q/k along one global axis."""
    if pairs == 0:
        return q, k
    width = pairs * 2
    sin, cos = _rope_sin_cos(coords, pairs, base, q.dtype)
    q_part = q[..., offset:offset + width]
    k_part = k[..., offset:offset + width]
    q = torch.cat([
        q[..., :offset], q_part * cos + _rotate_half(q_part) * sin,
        q[..., offset + width:]
    ], dim=-1)
    k = torch.cat([
        k[..., :offset], k_part * cos + _rotate_half(k_part) * sin,
        k[..., offset + width:]
    ], dim=-1)
    return q, k


def _apply_axis_rope_qk(q, k, q_coords, k_coords, offset, pairs, base=10000.0):
    """RoPE with possibly-different sequence lengths for q and k."""
    if pairs == 0:
        return q, k
    width = pairs * 2
    q_sin, q_cos = _rope_sin_cos(q_coords, pairs, base, q.dtype)
    k_sin, k_cos = _rope_sin_cos(k_coords, pairs, base, k.dtype)
    q_part = q[..., offset:offset + width]
    k_part = k[..., offset:offset + width]
    q = torch.cat([
        q[..., :offset], q_part * q_cos + _rotate_half(q_part) * q_sin,
        q[..., offset + width:]
    ], dim=-1)
    k = torch.cat([
        k[..., :offset], k_part * k_cos + _rotate_half(k_part) * k_sin,
        k[..., offset + width:]
    ], dim=-1)
    return q, k


class RoPE3DBottleneckAttention(nn.Module):
    """Cross-view spatio-temporal self-attention with global cylinder RoPE.

    Tokens from **all** virtual views participate in the same attention
    operation (no per-view isolation), and their RoPE coordinates come from a
    shared ego-cylinder coordinate system, so two tokens that look at the
    same physical azimuth/height get aligned RoPE phases regardless of which
    virtual view produced them. Combined with the global softmax, this is
    what enables redundant content across overlapping views to collapse into
    a single latent representation.

    To keep memory in check the attention is per-frame: tokens are packed as
    ``[B*T, V*H*W, C]``. Temporal mixing is handled by the surrounding 3D
    convolution blocks. RoPE is applied along (azimuth, height, time-in-clip)
    where the time axis uses a per-frame index broadcast across V*H*W.
    """

    def __init__(self, channels: int, num_heads: int,
                 rope_base: float = 10000.0):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.rope_base = rope_base

    def forward(self, x: torch.Tensor):
        # x: [B, T, V, C, H, W] -> tokens [B*T, V*H*W, C]
        b, t, v, c, h, w = x.shape
        tokens = x.permute(0, 1, 2, 4, 5, 3).reshape(b * t, v * h * w, c)
        qkv = self.qkv(self.norm(tokens))
        q, k, value = qkv.chunk(3, dim=-1)
        n = b * t
        q = q.view(n, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(n, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(
            n, -1, self.num_heads, self.head_dim).transpose(1, 2)

        az, hh = _build_view_coords(v, h, w, x.device, q.dtype)
        # Two axes: global azimuth and height. We don't include a temporal
        # coord here because attention is reshaped per-frame (B*T independent
        # batches), so all tokens in a frame share the same time index and a
        # time RoPE band would be a constant (wasting head_dim). Cross-frame
        # mixing is left to the surrounding causal-conv stack.
        coord_axes = [az, hh]
        pairs_per_axis = max(self.head_dim // 4, 1) \
            if self.head_dim >= 4 else 0
        offset = 0
        for coord in coord_axes:
            if offset + pairs_per_axis * 2 <= self.head_dim:
                q, k = _apply_axis_rope(
                    q, k, coord, offset, pairs_per_axis, base=self.rope_base)
                offset += pairs_per_axis * 2

        attn = F.scaled_dot_product_attention(q, k, value)
        attn = attn.transpose(1, 2).reshape(n, v * h * w, c)
        tokens = tokens + self.proj(attn)
        return tokens.reshape(b, t, v, h, w, c).permute(0, 1, 2, 5, 3, 4)\
            .contiguous()


class CrossView4DVAE(nn.Module):
    """4D VAE for compressing time, circular multi-view, and image space.

    Input and output use [B, T, V, C, H, W]. The model first projects input
    cameras onto a fixed circular virtual rig, then compresses view/time/space.
    """

    is_crossview_vae = True

    def __init__(self, **config):
        super().__init__()
        self.config = _namespace_config(config)
        cfg = self.config

        # Cylinder radii (in metres) at which to sample the virtual rig.
        # Defaults to a single mid-range radius that strikes a balance
        # between near-vehicle parallax and far-background alignment in
        # autonomous-driving scenes. Override via the ``cylinder_radii``
        # config key (e.g. ``[3.0, 10.0, 30.0]`` for multi-depth aggregation).
        cylinder_radii = getattr(cfg, "cylinder_radii", (10.0,))
        self.virtual_projector = CylindricalViewProjector(
            cfg.virtual_view_count, cylinder_radii=cylinder_radii)
        # Causal stem: time-causal 3D conv on the raw camera frames.
        self.stem = CausalConv3d(
            cfg.in_channels, cfg.base_channels, 3, padding=1)

        self.down1 = TimeSpaceDownBlock(
            cfg.base_channels, cfg.base_channels, stride_t=1, stride_hw=2)
        self.view_mix1 = ViewMixResBlock(cfg.base_channels)
        self.down2 = TimeSpaceDownBlock(
            cfg.base_channels, cfg.base_channels * 2, stride_t=2, stride_hw=2)
        self.view_mix2 = ViewMixResBlock(cfg.base_channels * 2)
        self.down3 = TimeSpaceDownBlock(
            cfg.base_channels * 2, cfg.base_channels * 4,
            stride_t=max(1, cfg.temporal_downsample_factor // 2),
            stride_hw=2)
        self.view_mix3 = ViewMixResBlock(cfg.base_channels * 4)

        # Pre-compression cross-view attention so redundant content from
        # overlapping virtual views aligns before the learnable view-query
        # compressor folds them down.
        self.pre_attn = nn.Sequential(*[
            RoPE3DBottleneckAttention(
                cfg.base_channels * 4, cfg.num_attention_heads)
            for _ in range(cfg.num_bottleneck_blocks)
        ])
        self.view_down = ViewQueryCompressor(
            cfg.base_channels * 4, cfg.num_attention_heads,
            cfg.latent_view_count)
        self.attn = nn.Sequential(*[
            RoPE3DBottleneckAttention(
                cfg.base_channels * 4, cfg.num_attention_heads)
            for _ in range(cfg.num_bottleneck_blocks)
        ])
        self.to_moments = nn.Linear(cfg.base_channels * 4,
                                    cfg.latent_channels * 2)

        self.from_latent = nn.Linear(cfg.latent_channels, cfg.base_channels * 4)
        self.view_up = ViewQueryExpander(
            cfg.base_channels * 4, cfg.num_attention_heads,
            cfg.virtual_view_count)
        self.post_attn = nn.Sequential(*[
            RoPE3DBottleneckAttention(
                cfg.base_channels * 4, cfg.num_attention_heads)
            for _ in range(cfg.num_bottleneck_blocks)
        ])
        self.up1 = TimeSpaceUpBlock(cfg.base_channels * 4, cfg.base_channels * 2)
        self.view_mix_up1 = ViewMixResBlock(cfg.base_channels * 2)
        self.up2 = TimeSpaceUpBlock(cfg.base_channels * 2, cfg.base_channels)
        self.view_mix_up2 = ViewMixResBlock(cfg.base_channels)
        self.up3 = TimeSpaceUpBlock(cfg.base_channels, cfg.base_channels)
        self.head = CausalConv3d(
            cfg.base_channels, cfg.out_channels, 3, padding=1)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, subfolder=None,
                        **kwargs):
        model_dir = pretrained_model_name_or_path
        if model_dir is not None and subfolder is not None:
            candidate = os.path.join(model_dir, subfolder)
            if os.path.isdir(candidate):
                model_dir = candidate

        config = {}
        if model_dir is not None and os.path.isdir(model_dir):
            for name in ("crossview_vae_config.json", "config.json"):
                path = os.path.join(model_dir, name)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        config.update(json.load(f))
                    break
        config.update(kwargs)
        model = cls(**config)

        if model_dir is not None and os.path.isdir(model_dir):
            for name in ("crossview_vae.pt", "pytorch_model.bin"):
                path = os.path.join(model_dir, name)
                if os.path.exists(path):
                    try:
                        model.load_state_dict(
                            torch.load(path, map_location="cpu"), strict=False)
                    except RuntimeError:
                        if name != "crossview_vae.pt":
                            continue
                        raise
                    break
        return model

    def _as_6d(self, x: torch.Tensor):
        if x.ndim == 6:
            return x, False
        if x.ndim == 5:
            # Fallback for video-only callers: [B, C, T, H, W].
            return x.permute(0, 2, 1, 3, 4).unsqueeze(2), True
        raise ValueError(
            "CrossView4DVAE expects [B,T,V,C,H,W] or [B,C,T,H,W].")

    def _stem(self, x: torch.Tensor):
        b, t, v, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1, 4, 5).reshape(b * v, c, t, h, w)
        y = self.stem(y)
        _, c2, t2, h2, w2 = y.shape
        return y.reshape(b, v, c2, t2, h2, w2).permute(0, 3, 1, 2, 4, 5)

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics_hw: Optional["tuple[int, int]"] = None,
    ):
        x, squeezed_view = self._as_6d(x)
        # If the caller did not specify the calibration resolution, default to
        # the *input* image resolution (i.e. the resolution at which the
        # batch was prepared). The projector itself further rescales from
        # this to its actual feature-map resolution.
        if intrinsics is not None and intrinsics_hw is None:
            intrinsics_hw = (int(x.shape[-2]), int(x.shape[-1]))
        # Run a cheap stem on raw cameras first, then re-bin onto the virtual
        # cylinder at the feature level (cheaper and more robust than doing
        # the geometric resample on RGB pixels).
        h = self._stem(x)
        h = self.virtual_projector(h, intrinsics, extrinsics, intrinsics_hw)
        h = self.down1(h)
        h = self.view_mix1(h)
        h = self.down2(h)
        h = self.view_mix2(h)
        h = self.down3(h)
        h = self.view_mix3(h)
        h = self.pre_attn(h)
        h = self.view_down(h)
        h = self.attn(h)
        moments = self.to_moments(h.permute(0, 1, 2, 4, 5, 3))
        mean, logvar = moments.chunk(2, dim=-1)
        mean = mean.permute(0, 1, 2, 5, 3, 4).contiguous()
        logvar = logvar.permute(0, 1, 2, 5, 3, 4).contiguous()
        if squeezed_view:
            mean = mean[:, :, 0].permute(0, 2, 1, 3, 4)
            logvar = logvar[:, :, 0].permute(0, 2, 1, 3, 4)
        output = EncoderOutput(DiagonalGaussianDistribution(mean, logvar))
        return output if return_dict else (output.latent_dist,)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        z, squeezed_view = self._as_6d(z)
        latent_time = z.shape[1]
        h = self.from_latent(z.permute(0, 1, 2, 4, 5, 3))
        h = h.permute(0, 1, 2, 5, 3, 4).contiguous()
        h = self.view_up(h, self.config.virtual_view_count)
        h = self.post_attn(h)
        h = self.up1(h, scale_t=max(1, self.config.temporal_downsample_factor // 2),
                     scale_hw=2)
        h = self.view_mix_up1(h)
        h = self.up2(h, scale_t=2, scale_hw=2)
        h = self.view_mix_up2(h)
        h = self.up3(h, scale_t=1, scale_hw=2)

        b, t, v, c, height, width = h.shape
        y = h.permute(0, 2, 3, 1, 4, 5).reshape(b * v, c, t, height, width)
        y = self.head(y)
        _, c2, t2, h2, w2 = y.shape
        y = y.reshape(b, v, c2, t2, h2, w2).permute(0, 3, 1, 2, 4, 5)
        target_time = self.config.temporal_pre + \
            max(0, latent_time - 1) * self.config.temporal_downsample_factor
        y = y[:, :target_time]
        if squeezed_view:
            y = y[:, :, 0].permute(0, 2, 1, 3, 4)
        if return_dict:
            return DecoderOutput(y)
        return (y,)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics_hw: Optional["tuple[int, int]"] = None,
    ):
        posterior = self.encode(
            x, intrinsics=intrinsics, extrinsics=extrinsics,
            intrinsics_hw=intrinsics_hw).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        reconstruction = self.decode(z).sample
        return {
            "sample": reconstruction,
            "posterior": posterior,
            "kl_loss": posterior.kl(),
        }
