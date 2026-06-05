import json
import math
import os
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
    config.setdefault("sequence_length", 5)
    config.setdefault("latent_time_count", 2)
    config.setdefault("near_latent_time_count", None)
    config.setdefault("far_latent_time_count", None)
    config.setdefault("near_latent_view_count", None)
    config.setdefault("far_latent_view_count", None)
    config.setdefault("temporal_downsample_factor", 4)
    config.setdefault("temporal_pre", 1)
    config.setdefault("spatial_downsample_factor", 8)
    config.setdefault("latent_spatial_downsample_factor", 1)
    config.setdefault("near_latent_spatial_downsample_factor", None)
    config.setdefault("far_latent_spatial_downsample_factor", None)
    config.setdefault("image_height", None)
    config.setdefault("image_width", None)
    config.setdefault("num_bottleneck_blocks", 2)
    config.setdefault("num_attention_heads", 2)
    config.setdefault("gradient_checkpoint_encode", False)
    config.setdefault("gradient_checkpoint_decode", False)
    config.setdefault("ego_coordinate_mode", "nuscenes")
    config.setdefault("decode_camera_from_cylinder", False)
    config.setdefault("cylinder_radii", (4.0, 20.0))
    config.setdefault("cylinder_height_scale", 1.0)
    config.setdefault("projector_edge_feather", 0.05)
    config.setdefault("projector_angular_power", 0.5)
    config.setdefault("projector_blend_mode", "soft")
    config.setdefault("cylinder_vertical_mode", "camera_fov")
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
        ego_coordinate_mode: str = "nuscenes",
        cylinder_height_scale: float = 1.0,
        edge_feather: float = 0.05,
        angular_power: float = 0.5,
        blend_mode: str = "soft",
        vertical_mode: str = "camera_fov",
    ):
        super().__init__()
        self.virtual_view_count = virtual_view_count
        if ego_coordinate_mode not in {"nuscenes", "synthetic"}:
            raise ValueError(
                "ego_coordinate_mode must be 'nuscenes' or 'synthetic', got "
                f"{ego_coordinate_mode!r}.")
        self.ego_coordinate_mode = ego_coordinate_mode
        self.cylinder_height_scale = float(cylinder_height_scale)
        self.edge_feather = float(edge_feather)
        self.angular_power = float(angular_power)
        if blend_mode not in {"soft", "best_camera"}:
            raise ValueError(
                "projector_blend_mode must be 'soft' or 'best_camera', got "
                f"{blend_mode!r}.")
        self.blend_mode = blend_mode
        if vertical_mode not in {"aspect", "camera_fov"}:
            raise ValueError(
                "cylinder_vertical_mode must be 'aspect' or 'camera_fov', got "
                f"{vertical_mode!r}.")
        self.vertical_mode = vertical_mode
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

    def _cylinder_y_extent(
        self,
        view_count: int,
        h: int,
        w: int,
        dtype,
        device,
        intrinsics: Optional[torch.Tensor] = None,
        intrinsics_hw: Optional["tuple[int, int]"] = None,
    ) -> torch.Tensor:
        """Full vertical cylinder range in height/radius units."""
        delta_phi = 2 * math.pi / view_count
        aspect_extent = delta_phi * (h / max(w, 1)) * self.cylinder_height_scale
        if self.vertical_mode != "camera_fov" or intrinsics is None:
            return torch.as_tensor(aspect_extent, dtype=dtype, device=device)

        fy = intrinsics[..., 1, 1].float().clamp(min=1e-6)
        if intrinsics_hw is not None:
            h_cal = float(intrinsics_hw[0])
            full_extent = h_cal / fy
        else:
            cy = intrinsics[..., 1, 2].float().clamp(min=1e-6)
            full_extent = (2.0 * cy) / fy
        # Use a high quantile instead of the median. The camera render path
        # must cover the tallest camera FOV in the rig; otherwise valid camera
        # pixels hit outside the cylinder and appear as gray curved bands.
        # Keeping a quantile rather than max avoids one badly calibrated view
        # blowing up the canonical cylinder height for every sample.
        extent = torch.quantile(full_extent.flatten(), 0.95)
        scale = max(self.cylinder_height_scale, 1e-3)
        extent = extent * scale
        # The aspect-ratio heuristic is still a useful lower bound for low-res
        # feature maps and synthetic data.
        aspect_extent = torch.as_tensor(
            aspect_extent, dtype=torch.float32, device=extent.device)
        extent = torch.maximum(extent, aspect_extent)
        return extent.to(dtype=dtype, device=device)

    @staticmethod
    def _virtual_azimuth(virtual_view_count: int, device, dtype):
        # Returns the azimuth (rad) of the optical axis of each virtual view,
        # uniformly distributed around 360 deg.
        return (torch.arange(virtual_view_count, device=device, dtype=dtype)
                + 0.5) * (2 * math.pi / virtual_view_count) - math.pi

    def _points_from_phi_height(
        self,
        phi: torch.Tensor,
        height_unit: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        """Map cylinder coordinates to ego points.

        ``nuscenes`` uses the common AV convention: x/y is the ground plane
        and z is up. ``synthetic`` keeps the original toy-data convention:
        x/z is the ground plane and y is up.
        """
        r = float(radius)
        if self.ego_coordinate_mode == "nuscenes":
            x = torch.cos(phi) * r
            y = torch.sin(phi) * r
            z = height_unit * r
            return torch.stack([x, y, z], dim=-1)
        x = torch.sin(phi) * r
        y = height_unit * r
        z = torch.cos(phi) * r
        return torch.stack([x, y, z], dim=-1)

    def _phi_height_from_points(self, points: torch.Tensor, radius: float):
        r = float(radius)
        if self.ego_coordinate_mode == "nuscenes":
            phi = torch.atan2(points[..., 1], points[..., 0])
            height_unit = points[..., 2] / max(r, 1e-6)
        else:
            phi = torch.atan2(points[..., 0], points[..., 2])
            height_unit = points[..., 1] / max(r, 1e-6)
        return phi, height_unit

    def _ray_project(
        self,
        x: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics_hw: Optional["tuple[int, int]"] = None,
        return_layers: bool = False,
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
        y_extent = self._cylinder_y_extent(
            v_out, h, w, dtype, device, intrinsics, intrinsics_hw)
        y_grid = (-v_lin).view(1, h, 1) * y_extent
        y_grid = y_grid.expand(v_out, h, w)
        # Cylinder coordinates in ego frame.
        phi = phi.expand(v_out, h, w)
        cyl_unit = self._points_from_phi_height(
            phi, y_grid, radius=1.0)  # [V_out, H, W, 3]

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

        cyl_unit_flat = cyl_unit.reshape(-1, 3)                    # [N, 3]
        feat = x.reshape(b * t * v_in, c, h, w)

        # Multi-depth aggregation: cast rays at each ``cylinder_radius``,
        # weighted-average the resulting samples. With a single radius this
        # collapses to a plain finite-depth projection. Using multiple radii
        # lets the network pick whichever depth aligns best at each spatial
        # location.
        out_acc = None
        weight_acc = None
        layer_outs: list[torch.Tensor] = []
        for r in self.cylinder_radii.tolist():
            # 3D points on the cylinder of radius r in the ego frame.
            points_ego = cyl_unit_flat * r                       # [N, 3]
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
            # OpenCV-style intrinsics use integer pixel centers. For
            # grid_sample(..., align_corners=False), pixel center u maps to
            # normalized coordinate 2 * (u + 0.5) / W - 1.
            gx = 2.0 * (u_img + 0.5) / max(w, 1) - 1
            gy = 2.0 * (v_img + 0.5) / max(h, 1) - 1
            in_image = (gx.abs() < 1) & (gy.abs() < 1) & valid_front
            border_dist = torch.minimum(1.0 - gx.abs(), 1.0 - gy.abs())
            border_dist = border_dist.clamp(0, 1)
            if self.edge_feather > 0:
                t_edge = (border_dist / self.edge_feather).clamp(0, 1)
                edge_feather = t_edge * t_edge * (3.0 - 2.0 * t_edge)
            else:
                edge_feather = torch.ones_like(gx)
            # Keep more confidence near image centers and smoothly fade near
            # borders, which avoids hard source changes at camera overlaps.
            border_weight = edge_feather * border_dist.clamp(min=1e-4).sqrt()
            center_radius = (gx.square() + gy.square()).sqrt()
            center_sigma = 0.55
            center_weight = torch.exp(
                -0.5 * (center_radius / center_sigma).square())

            # Visibility weight: angle between (camera -> point) vector and
            # the camera's principal axis in the ego frame, masked by FOV.
            # Using the proper from-camera vector (rather than the from-ego
            # ray direction) gives a more accurate FOV mask when the camera
            # is offset from the ego origin.
            vec = points_ego.view(1, 1, 1, -1, 3) \
                - cam_pos_ego.unsqueeze(-2)                      # [B,T,V_in,N,3]
            vec_n = vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            cos_axis = (vec_n * cam_z_ego.unsqueeze(-2)).sum(-1)  # [B,T,V_in,N]
            theta = torch.acos(cos_axis.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
            sigma = max(self.fov / 3.0, 1e-6)
            angular = torch.exp(-0.5 * (theta / sigma).square())
            angular = angular * (theta <= (self.fov / 2)).to(angular.dtype)
            if self.angular_power > 0:
                angular = angular.pow(self.angular_power)
            raw_weight = angular * border_weight * center_weight
            valid_weight = in_image & (angular > 0)
            weight_logits = torch.where(
                valid_weight,
                raw_weight.clamp(min=1e-12).log(),
                torch.full_like(raw_weight, -1e4),
            )
            weight = torch.softmax(weight_logits, dim=2)
            weight = weight * valid_weight.to(weight.dtype)

            # Sample features from each input camera at this depth.
            grid = torch.stack([gx, gy], dim=-1).reshape(
                b * t * v_in, v_out, h * w, 2)
            sampled = F.grid_sample(
                feat, grid, mode="bilinear", padding_mode="border",
                align_corners=False)
            sampled = sampled.view(b, t, v_in, c, v_out, h, w)
            weight = weight.view(b, t, v_in, 1, v_out, h, w)
            if self.blend_mode == "best_camera":
                # Pick one dominant physical camera per virtual view. A
                # per-pixel argmax removes averaging ghosts, but it creates
                # hard rectangular source-switching bands inside the same
                # cylinder slice. View-level selection keeps the RGB target
                # coherent while still using geometry to route each slice.
                score = weight.mean(dim=(-1, -2), keepdim=True)
                best = score.argmax(dim=2, keepdim=True)
                best = best.expand(b, t, 1, 1, v_out, h, w)
                keep = torch.zeros_like(weight).scatter_(2, best, 1.0)
                weight = weight * keep

            weighted = sampled * weight
            layer_weight = weight.sum(dim=2).clamp(min=1e-4)
            layer_out = weighted.sum(dim=2) / layer_weight
            layer_outs.append(
                layer_out.permute(0, 1, 3, 2, 4, 5).contiguous())

            if return_layers:
                continue

            if out_acc is None:
                out_acc = weighted
                weight_acc = weight
            else:
                out_acc = out_acc + weighted
                weight_acc = weight_acc + weight

        if return_layers:
            # [B,T,D,V_out,C,H,W], where D indexes the configured radii.
            return torch.stack(layer_outs, dim=2)

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

    def project_layers(
        self,
        x: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics_hw: Optional["tuple[int, int]"] = None,
    ) -> torch.Tensor:
        """Project cameras to a true multi-layer cylinder.

        Output layout is ``[B,T,D,V,C,H,W]`` where ``D`` indexes
        ``self.cylinder_radii``. Unlike ``forward()``, this does not collapse
        radii into a single average, which preserves near/far information for
        the VAE bottleneck.
        """
        if intrinsics is not None and extrinsics is not None:
            return self._ray_project(
                x, intrinsics, extrinsics, intrinsics_hw,
                return_layers=True)
        base = self.forward(x, None, None, None)
        d = int(self.cylinder_radii.numel())
        return base.unsqueeze(2).expand(-1, -1, d, -1, -1, -1, -1).contiguous()

    def render_cylinder_to_cameras(
        self,
        cylinder: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics_hw: Optional["tuple[int, int]"] = None,
        target_hw: Optional["tuple[int, int]"] = None,
        layer_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Render canonical cylinder features back to physical camera grids.

        This is the inverse path needed for camera-domain reconstruction loss:
        decoder predicts a canonical cylinder, then each target camera samples
        from it using its K/E. It preserves the model's variable-camera
        canonical bottleneck while keeping the loss in raw camera space.
        """
        if cylinder.ndim == 6:
            cylinder_layers = cylinder.unsqueeze(2)
        elif cylinder.ndim == 7:
            cylinder_layers = cylinder
        else:
            raise ValueError(
                "render_cylinder_to_cameras expects [B,T,V,C,H,W] or "
                f"[B,T,D,V,C,H,W], got shape={tuple(cylinder.shape)}")
        b, t, d_cyl, v_cyl, c, h_cyl, w_cyl = cylinder_layers.shape
        if target_hw is None:
            h_out, w_out = h_cyl, w_cyl
        else:
            h_out, w_out = int(target_hw[0]), int(target_hw[1])
        device = cylinder.device
        dtype = cylinder.dtype

        if intrinsics.dim() == 4:
            v_cam = intrinsics.shape[1]
            intrinsics = intrinsics.unsqueeze(1).expand(b, t, v_cam, 3, 3)
        else:
            v_cam = intrinsics.shape[2]
        if extrinsics.dim() == 4:
            extrinsics = extrinsics.unsqueeze(1).expand(b, t, v_cam, 4, 4)
        if layer_weights is not None:
            if layer_weights.ndim != 7:
                raise ValueError(
                    "layer_weights must be [B,T,D,V_cam,1,H,W], got "
                    f"{tuple(layer_weights.shape)}")
            if layer_weights.shape[:4] != (b, t, d_cyl, v_cam):
                raise ValueError(
                    "layer_weights leading dims must match "
                    f"[B,T,D,V_cam]={b,t,d_cyl,v_cam}, got "
                    f"{tuple(layer_weights.shape[:4])}")
            if layer_weights.shape[4] != 1:
                raise ValueError("layer_weights channel dimension must be 1.")
            if layer_weights.shape[-2:] != (h_out, w_out):
                lw = layer_weights.reshape(b * t * d_cyl * v_cam, 1,
                                           *layer_weights.shape[-2:])
                lw = F.interpolate(
                    lw, size=(h_out, w_out), mode="bilinear",
                    align_corners=False)
                layer_weights = lw.reshape(b, t, d_cyl, v_cam, 1, h_out, w_out)
            layer_weights = layer_weights.to(device=device, dtype=dtype)

        if intrinsics_hw is not None:
            h_cal = float(intrinsics_hw[0])
            w_cal = float(intrinsics_hw[1])
            sx = w_out / max(w_cal, 1.0)
            sy = h_out / max(h_cal, 1.0)
            fx = intrinsics[..., 0, 0:1] * sx
            fy = intrinsics[..., 1, 1:2] * sy
            cx = intrinsics[..., 0, 2:3] * sx
            cy = intrinsics[..., 1, 2:3] * sy
        else:
            cx_raw = intrinsics[..., 0, 2:3]
            cy_raw = intrinsics[..., 1, 2:3]
            sx = w_out / (2.0 * cx_raw.clamp(min=1e-3))
            sy = h_out / (2.0 * cy_raw.clamp(min=1e-3))
            fx = intrinsics[..., 0, 0:1] * sx
            fy = intrinsics[..., 1, 1:2] * sy
            cx = cx_raw * sx
            cy = cy_raw * sy

        yy, xx = torch.meshgrid(
            torch.arange(h_out, device=device, dtype=dtype) + 0.5,
            torch.arange(w_out, device=device, dtype=dtype) + 0.5,
            indexing="ij",
        )
        xx = xx.view(1, 1, 1, h_out, w_out)
        yy = yy.view(1, 1, 1, h_out, w_out)
        x_cam = (xx - cx[..., None]) / fx[..., None].clamp(min=1e-6)
        y_cam = (yy - cy[..., None]) / fy[..., None].clamp(min=1e-6)
        z_cam = torch.ones_like(x_cam)
        dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        rot = extrinsics[..., :3, :3].to(dtype)
        origin = extrinsics[..., :3, 3].to(dtype)
        dirs_ego = torch.einsum("btvij,btvhwj->btvhwi", rot, dirs_cam)

        if self.ego_coordinate_mode == "nuscenes":
            h0 = 0
            h1 = 1
        else:
            h0 = 0
            h1 = 2

        y_extent = self._cylinder_y_extent(
            v_cyl, h_cyl, w_cyl, dtype, device, intrinsics, intrinsics_hw)
        out_acc = cylinder.new_zeros(b, t, v_cam, c, h_out, w_out)
        weight_acc = cylinder.new_zeros(b, t, v_cam, 1, h_out, w_out)

        origin_h0 = origin[..., h0].unsqueeze(-1).unsqueeze(-1)
        origin_h1 = origin[..., h1].unsqueeze(-1).unsqueeze(-1)
        dir_h0 = dirs_ego[..., h0]
        dir_h1 = dirs_ego[..., h1]
        radii = self.cylinder_radii.tolist()
        if len(radii) < d_cyl:
            radii = radii + [radii[-1]] * (d_cyl - len(radii))
        for layer_idx, r in enumerate(radii[:d_cyl]):
            cylinder = cylinder_layers[:, :, layer_idx]
            rr = float(r)
            a = dir_h0.square() + dir_h1.square()
            bq = 2.0 * (origin_h0 * dir_h0 + origin_h1 * dir_h1)
            cq = origin_h0.square() + origin_h1.square() - rr * rr
            disc = bq.square() - 4.0 * a * cq
            valid = (disc > 0) & (a.abs() > 1e-8)
            sqrt_disc = disc.clamp(min=0).sqrt()
            s1 = (-bq - sqrt_disc) / (2.0 * a.clamp(min=1e-8))
            s2 = (-bq + sqrt_disc) / (2.0 * a.clamp(min=1e-8))
            big = torch.full_like(s1, 1e8)
            s = torch.where(s1 > 1e-4, s1, torch.where(s2 > 1e-4, s2, big))
            valid = valid & (s < 1e7)

            pts = origin.unsqueeze(-2).unsqueeze(-2) + dirs_ego * s.unsqueeze(-1)
            phi, height_unit = self._phi_height_from_points(pts, rr)
            # Sample the decoded cylinder as one continuous circular panorama.
            # A physical camera FOV can span multiple virtual-view slices; hard
            # switching at ``view_idx`` creates visible vertical seams in the
            # rendered camera image. Circular padding lets bilinear sampling
            # blend across both internal slice boundaries and the wraparound.
            u_cyl = torch.remainder((phi + math.pi) / (2 * math.pi), 1.0)
            gy = ((-height_unit / max(y_extent, 1e-6)) + 0.5) * 2.0 - 1.0
            valid = valid & (gy.abs() <= 1.0)

            pano = cylinder.permute(0, 1, 3, 4, 2, 5).reshape(
                b * t, c, h_cyl, v_cyl * w_cyl)
            pad = 1
            pano = F.pad(pano, (pad, pad, 0, 0), mode="circular")
            pano_w = v_cyl * w_cyl
            padded_w = pano_w + 2 * pad
            # Convert circular [0,1) coordinate to align_corners=False grid
            # coordinates on the padded panorama.
            gx = 2.0 * (u_cyl * pano_w + pad + 0.5) / padded_w - 1.0
            grid = torch.stack([gx, gy], dim=-1).reshape(
                b * t * v_cam, h_out, w_out, 2)
            inp = pano.repeat_interleave(v_cam, dim=0)
            sampled = F.grid_sample(
                inp, grid, mode="bilinear", padding_mode="zeros",
                align_corners=False)
            sampled = sampled.view(b, t, v_cam, c, h_out, w_out)
            m = valid.unsqueeze(3).to(dtype)
            if layer_weights is not None:
                m = m * layer_weights[:, :, layer_idx]
            out_acc = out_acc + sampled * m
            weight_acc = weight_acc + m

        return out_acc / weight_acc.clamp(min=1e-4)


class CircularConv3d(nn.Module):
    """3D conv whose first spatial axis is the circular (panoramic) view axis.

    Input layout: ``[N, C, V, H, W]``. The view axis is wrapped with circular
    padding so the convolution treats the multi-view rig as a closed ring.
    The in-slice W axis is also circular padded to avoid zero-padding seams
    at angular boundaries.
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
        if pw:
            x = F.pad(x, (pw, pw, 0, 0, 0, 0), mode="circular")
        if ph:
            x = F.pad(x, (0, 0, ph, ph, 0, 0))
        if pv:
            x = F.pad(x, (0, 0, 0, 0, pv, pv), mode="circular")
        return self.conv(x)


class CausalConv3d(nn.Conv3d):
    """3D convolution with causal padding on the time axis (Wan2.1 style).

    Input layout: ``[N, C, T, H, W]``. The full temporal padding budget
    ``2 * (kernel_t // 2)`` is pushed to the **left** so that ``output[t]``
    only depends on ``input[<= t]`` (after stride). H uses standard symmetric
    padding. W can optionally use circular padding for cylinder panoramas.

    Like Wan2.1's CausalConv3d, the layer accepts an optional ``cache_x``
    argument: at autoregressive / streaming inference time the caller passes
    the last few frames from the previous chunk so this chunk's output is
    bit-identical to the non-streamed version. This is what makes the VAE
    safe to pair with a diffusion model that generates frames step by step.
    """

    def __init__(self, *args, **kwargs):
        self.circular_w = bool(kwargs.pop("circular_w", False))
        super().__init__(*args, **kwargs)
        # nn.Conv3d.padding is (pT, pH, pW). Move the full temporal budget
        # to the left and zero out the temporal padding inside Conv3d itself.
        pT, pH, pW = self.padding
        self._causal_padding = (int(pW), int(pH), int(2 * pT), 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None):
        p_w, p_h, p_t_left, p_t_right = self._causal_padding
        if cache_x is not None and p_t_left > 0:
            cache_x = cache_x.to(x.device, dtype=x.dtype)
            x = torch.cat([cache_x, x], dim=2)
            p_t_left = max(p_t_left - cache_x.shape[2], 0)
        if self.circular_w and p_w > 0:
            x = F.pad(x, (p_w, p_w, 0, 0, 0, 0), mode="circular")
            p_w = 0
        x = F.pad(x, (p_w, p_w, p_h, p_h, p_t_left, p_t_right))
        return super().forward(x)


def _build_view_coords(view_count: int, h: int, w: int, device, dtype,
                       ref_view_count: Optional[int] = None,
                       ref_h: Optional[int] = None,
                       ref_w: Optional[int] = None):
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
    if ref_h is None:
        ref_h = h
    if ref_w is None:
        ref_w = w
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
    px_per_view = ref_view_count * ref_w / float(view_count)
    v_idx = torch.arange(view_count, device=device, dtype=dtype)
    u_idx = torch.arange(w, device=device, dtype=dtype) + 0.5
    az = v_idx.view(view_count, 1, 1) * px_per_view \
        + u_idx.view(1, 1, w) * (px_per_view / w)
    az = az.expand(view_count, h, w).reshape(-1)
    y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * (
        float(ref_h) / float(max(h, 1)))
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


def _to_cylinder_pano(x: torch.Tensor) -> torch.Tensor:
    # [B,T,V,C,H,W] -> [B,C,T,H,V*W], with the full angular axis contiguous.
    return x.permute(0, 3, 1, 4, 2, 5).reshape(
        x.shape[0], x.shape[3], x.shape[1], x.shape[4],
        x.shape[2] * x.shape[5])


def _from_cylinder_pano(y: torch.Tensor, view_count: int) -> torch.Tensor:
    # [B,C,T,H,V*W] -> [B,T,V,C,H,W].
    b, c, t, h, pano_w = y.shape
    if pano_w % view_count != 0:
        raise ValueError(
            f"Panorama width {pano_w} is not divisible by view_count "
            f"{view_count}.")
    w = pano_w // view_count
    return y.reshape(b, c, t, h, view_count, w).permute(
        0, 2, 4, 1, 3, 5).contiguous()


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
            stride=(stride_t, stride_hw, stride_hw), padding=1,
            circular_w=True)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = CausalConv3d(
            out_channels, out_channels, 3, padding=1, circular_w=True)
        self.skip = CausalConv3d(
            channels, out_channels, 1,
            stride=(stride_t, stride_hw, stride_hw), padding=0,
            circular_w=True)

    def forward(self, x: torch.Tensor):
        # x: [B, T, V, C, H, W]
        v = int(x.shape[2])
        y = _to_cylinder_pano(x)
        skip = self.skip(y)
        h_y = self.conv1(F.silu(_per_frame_groupnorm(self.norm1, y)))
        h_y = self.conv2(F.silu(_per_frame_groupnorm(self.norm2, h_y)))
        y = h_y + skip
        return _from_cylinder_pano(y, v)


class TimeSpaceUpBlock(nn.Module):
    """Per-view residual block that upsamples (T, H, W) causally.

    The temporal upsample is implemented as nearest-neighbour repeat (also
    causal) followed by causal 3D convolutions, mirroring Wan2.1's decoder.
    """

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = CausalConv3d(
            channels, out_channels, 3, padding=1, circular_w=True)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = CausalConv3d(
            out_channels, out_channels, 3, padding=1, circular_w=True)

    def forward(self, x: torch.Tensor, scale_t: int, scale_hw: int):
        v = int(x.shape[2])
        y = _to_cylinder_pano(x)
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
        return _from_cylinder_pano(y, v)


class ViewMixResBlock(nn.Module):
    """Per-frame residual block on the continuous cylinder panorama.

    The physical angular axis is ``V * W``, not two independent circular axes.
    Running the convolution on the flattened panorama avoids a slice-local W
    wrap, which would otherwise connect the right edge of a virtual view back
    to its own left edge instead of to the next virtual view.
    """

    def __init__(self, channels: int, view_kernel: int = 3,
                 spatial_kernel: int = 3):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.pad_h = spatial_kernel // 2
        self.pad_w = max(view_kernel, spatial_kernel) // 2
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=(spatial_kernel, max(view_kernel, spatial_kernel)),
            padding=0)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=(spatial_kernel, max(view_kernel, spatial_kernel)),
            padding=0)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_w:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        if self.pad_h:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, V, C, H, W]
        b, t, v, c, h, w = x.shape
        y = x.permute(0, 1, 3, 4, 2, 5).reshape(b * t, c, h, v * w)
        h_y = self.conv1(self._pad(F.silu(self.norm1(y))))
        h_y = self.conv2(self._pad(F.silu(self.norm2(h_y))))
        y = y + h_y
        return y.reshape(b, t, c, h, v, w).permute(0, 1, 4, 2, 3, 5)\
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

    def _ddp_identity(self, x: torch.Tensor) -> torch.Tensor:
        # Keep parameters in the autograd graph when identity mode is active,
        # so DistributedDataParallel does not report unused parameters.
        anchor = x.new_zeros(())
        for p in self.parameters():
            anchor = anchor + p.sum() * 0.0
        return x + anchor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, V_in, C, H, W]
        b, t, v_in, c, h, w = x.shape
        v_out = self.latent_views
        if v_out == v_in:
            return self._ddp_identity(x)
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

    def _ddp_identity(self, x: torch.Tensor) -> torch.Tensor:
        # Keep parameters in the autograd graph when identity mode is active,
        # so DistributedDataParallel does not report unused parameters.
        anchor = x.new_zeros(())
        for p in self.parameters():
            anchor = anchor + p.sum() * 0.0
        return x + anchor

    def forward(self, x: torch.Tensor, target_views: Optional[int] = None
                ) -> torch.Tensor:
        b, t, v_in, c, h, w = x.shape
        v_out = target_views if target_views is not None else self.virtual_views
        if v_out > self.virtual_views:
            raise ValueError(
                f"ViewQueryExpander built for at most {self.virtual_views} "
                f"virtual views, got target_views={v_out}.")
        if v_out == v_in:
            return self._ddp_identity(x)
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


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


class TVJointViewCompressor(nn.Module):
    """Layer-local compressor that jointly reduces time, view and space tokens.

    The old global version let every latent spatial token attend to every
    source spatial token, which made the KV sequence ``T*V*H*W``. This version
    first maps the source to the query spatial grid, then runs attention per
    physical spatial column. Each column only sees its own ``T*V`` tokens.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        latent_times: int,
        latent_views: int,
        latent_spatial_downsample: int = 1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.latent_times = int(latent_times)
        self.latent_views = int(latent_views)
        self.latent_spatial_downsample = int(latent_spatial_downsample)
        if self.latent_spatial_downsample < 1:
            raise ValueError(
                "latent_spatial_downsample must be >= 1, got "
                f"{self.latent_spatial_downsample}.")
        self.time_query_emb = nn.Parameter(
            torch.randn(self.latent_times, channels) * 0.02)
        self.view_query_emb = nn.Parameter(
            torch.randn(self.latent_views, channels) * 0.02)
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(channels, channels * 2)
        self.proj = nn.Linear(channels, channels)
        self.rope_base = rope_base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,C,H,W]
        b, t, v_in, c, h, w = x.shape
        t_out = self.latent_times
        v_out = self.latent_views
        h_out = _ceil_div(h, self.latent_spatial_downsample)
        w_out = _ceil_div(w, self.latent_spatial_downsample)
        if (h_out, w_out) == (h, w):
            x_ctx = x
        else:
            x_pool = x.reshape(b * t * v_in, c, h, w)
            x_pool = F.adaptive_avg_pool2d(x_pool, (h_out, w_out))
            x_ctx = x_pool.reshape(b, t, v_in, c, h_out, w_out)
        mean = x_ctx.mean(dim=(1, 2), keepdim=True)
        q_seed = mean.expand(-1, t_out, v_out, -1, -1, -1)
        q_seed = q_seed + self.time_query_emb.view(
            1, t_out, 1, c, 1, 1)
        q_seed = q_seed + self.view_query_emb.view(
            1, 1, v_out, c, 1, 1)

        q_tokens = q_seed.permute(0, 4, 5, 1, 2, 3).reshape(
            b * h_out * w_out, t_out * v_out, c)
        kv_tokens = x_ctx.permute(0, 4, 5, 1, 2, 3).reshape(
            b * h_out * w_out, t * v_in, c)

        q = self.to_q(self.norm_q(q_tokens))
        kv = self.to_kv(self.norm_kv(kv_tokens))
        k, v_feat = kv.chunk(2, dim=-1)

        bhw = b * h_out * w_out
        q = q.view(bhw, t_out * v_out, self.num_heads, self.head_dim)\
            .transpose(1, 2)
        k = k.view(bhw, t * v_in, self.num_heads, self.head_dim)\
            .transpose(1, 2)
        v_feat = v_feat.view(
            bhw, t * v_in, self.num_heads, self.head_dim).transpose(1, 2)

        ref_v = max(v_in, v_out)
        q_az, _ = _build_view_coords(
            v_out, 1, 1, x.device, q.dtype, ref_view_count=ref_v)
        k_az, _ = _build_view_coords(
            v_in, 1, 1, x.device, k.dtype, ref_view_count=ref_v)
        q_az = q_az.repeat(t_out)
        k_az = k_az.repeat(t)
        q_time = torch.linspace(
            0, max(t - 1, 0), t_out, device=x.device, dtype=q.dtype)\
            .repeat_interleave(v_out)
        k_time = torch.arange(t, device=x.device, dtype=k.dtype)\
            .repeat_interleave(v_in)

        pairs = max(self.head_dim // 4, 1) if self.head_dim >= 4 else 0
        offset = 0
        if pairs:
            q, k = _apply_axis_rope_qk(
                q, k, q_time, k_time, offset, pairs, base=self.rope_base)
            offset += pairs * 2
            if offset + pairs * 2 <= self.head_dim:
                q, k = _apply_axis_rope_qk(
                    q, k, q_az, k_az, offset, pairs, base=self.rope_base)

        attn = F.scaled_dot_product_attention(q, k, v_feat)
        attn = attn.transpose(1, 2).reshape(bhw, t_out * v_out, c)
        out = q_tokens + self.proj(attn)
        out = out.reshape(b, h_out, w_out, t_out, v_out, c).permute(
            0, 3, 4, 1, 2, 5)
        out = out.reshape(b, t_out * v_out * h_out * w_out, c)
        return out[:, None, :, :, None, None].contiguous()


class TVJointViewExpander(nn.Module):
    """Mirror of ``TVJointViewCompressor`` for one cylinder layer.

    Latent spatial grids are upsampled first, then each output spatial column
    attends only to that column's latent ``T*V`` tokens.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        latent_times: int,
        latent_views: int,
        virtual_views: int,
        latent_spatial_downsample: int = 1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.latent_times = int(latent_times)
        self.latent_views = int(latent_views)
        self.virtual_views = int(virtual_views)
        self.latent_spatial_downsample = int(latent_spatial_downsample)
        if self.latent_spatial_downsample < 1:
            raise ValueError(
                "latent_spatial_downsample must be >= 1, got "
                f"{self.latent_spatial_downsample}.")
        self.time_query_emb = nn.Parameter(torch.randn(1, channels) * 0.02)
        self.view_query_emb = nn.Parameter(
            torch.randn(self.virtual_views, channels) * 0.02)
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(channels, channels * 2)
        self.proj = nn.Linear(channels, channels)
        self.rope_base = rope_base

    def forward(
        self,
        x: torch.Tensor,
        target_time: int,
        target_views: Optional[int] = None,
        target_hw: Optional[tuple[int, int]] = None,
                ) -> torch.Tensor:
        # x: [B,1,T_lat*V_lat*H_lat*W_lat,C,1,1]
        b, _t_pkg, token_count, c, h_pkg, w_pkg = x.shape
        if target_hw is None:
            raise ValueError(
                "target_hw is required for packed spatial latent decoding.")
        h_out, w_out = int(target_hw[0]), int(target_hw[1])
        h_in = _ceil_div(h_out, self.latent_spatial_downsample)
        w_in = _ceil_div(w_out, self.latent_spatial_downsample)
        t_in = self.latent_times
        v_in = self.latent_views
        expected = t_in * v_in * h_in * w_in
        if token_count != expected:
            raise ValueError(
                "Layer latent token count mismatch: "
                f"got {token_count}, expected {expected} "
                f"({t_in} time x {v_in} views x {h_in}x{w_in} spatial).")
        if (h_pkg, w_pkg) != (1, 1):
            raise ValueError(
                "Packed spatial latents must use trailing spatial shape 1x1, "
                f"got {h_pkg}x{w_pkg}.")
        x = x[:, 0, :, :, 0, 0].reshape(b, t_in, v_in, h_in, w_in, c)
        t_out = int(target_time)
        v_out = int(target_views) if target_views is not None else self.virtual_views
        if v_out > self.virtual_views:
            raise ValueError(
                f"TVJointViewExpander built for at most {self.virtual_views} "
                f"views, got target_views={v_out}.")
        if (h_in, w_in) == (h_out, w_out):
            x_ctx = x
        else:
            x_up = x.permute(0, 1, 2, 5, 3, 4).reshape(
                b * t_in * v_in, c, h_in, w_in)
            x_up = F.interpolate(
                x_up, size=(h_out, w_out), mode="bilinear",
                align_corners=False)
            x_ctx = x_up.reshape(
                b, t_in, v_in, c, h_out, w_out).permute(
                    0, 1, 2, 4, 5, 3)
        mean = x_ctx.mean(dim=(1, 2), keepdim=True)
        q_seed = mean.expand(-1, t_out, v_out, -1, -1, -1)
        q_seed = q_seed + self.time_query_emb.view(1, 1, 1, 1, 1, c)
        q_seed = q_seed + self.view_query_emb[:v_out].view(
            1, 1, v_out, 1, 1, c)

        q_tokens = q_seed.permute(0, 3, 4, 1, 2, 5).reshape(
            b * h_out * w_out, t_out * v_out, c)
        kv_tokens = x_ctx.permute(0, 3, 4, 1, 2, 5).reshape(
            b * h_out * w_out, t_in * v_in, c)

        q = self.to_q(self.norm_q(q_tokens))
        kv = self.to_kv(self.norm_kv(kv_tokens))
        k, v_feat = kv.chunk(2, dim=-1)

        bhw = b * h_out * w_out
        q = q.view(bhw, t_out * v_out, self.num_heads, self.head_dim)\
            .transpose(1, 2)
        k = k.view(bhw, t_in * v_in, self.num_heads, self.head_dim)\
            .transpose(1, 2)
        v_feat = v_feat.view(
            bhw, t_in * v_in, self.num_heads, self.head_dim).transpose(1, 2)

        ref_v = max(v_in, v_out)
        q_az, _ = _build_view_coords(
            v_out, 1, 1, x.device, q.dtype, ref_view_count=ref_v)
        k_az, _ = _build_view_coords(
            v_in, 1, 1, x.device, k.dtype, ref_view_count=ref_v)
        q_az = q_az.repeat(t_out)
        k_az = k_az.repeat(t_in)
        q_time = torch.arange(t_out, device=x.device, dtype=q.dtype)\
            .repeat_interleave(v_out)
        k_time = torch.linspace(
            0, max(t_out - 1, 0), t_in, device=x.device, dtype=k.dtype)\
            .repeat_interleave(v_in)

        pairs = max(self.head_dim // 4, 1) if self.head_dim >= 4 else 0
        offset = 0
        if pairs:
            q, k = _apply_axis_rope_qk(
                q, k, q_time, k_time, offset, pairs, base=self.rope_base)
            offset += pairs * 2
            if offset + pairs * 2 <= self.head_dim:
                q, k = _apply_axis_rope_qk(
                    q, k, q_az, k_az, offset, pairs, base=self.rope_base)

        attn = F.scaled_dot_product_attention(q, k, v_feat)
        attn = attn.transpose(1, 2).reshape(bhw, t_out * v_out, c)
        out = q_tokens + self.proj(attn)
        out = out.reshape(b, h_out, w_out, t_out, v_out, c).permute(
            0, 3, 4, 5, 1, 2)
        return out.contiguous()


class LayeredTVCompressor(nn.Module):
    """Compress each cylinder layer with its own token budget."""

    def __init__(self, channels: int, num_heads: int,
                 latent_shapes: list[tuple[int, int]],
                 spatial_factors: list[int]):
        super().__init__()
        self.latent_shapes = [(int(t), int(v)) for t, v in latent_shapes]
        self.spatial_factors = [int(s) for s in spatial_factors]
        if len(self.spatial_factors) != len(self.latent_shapes):
            raise ValueError(
                "spatial_factors must match latent_shapes length: "
                f"{len(self.spatial_factors)} vs {len(self.latent_shapes)}")
        self.last_source_hw: Optional[tuple[int, int]] = None
        self.last_latent_spatial_shapes: list[tuple[int, int]] = []
        self.last_latent_token_counts: list[int] = []
        self.layers = nn.ModuleList([
            TVJointViewCompressor(channels, num_heads, t, v, s)
            for (t, v), s in zip(self.latent_shapes, self.spatial_factors)
        ])

    @property
    def total_latent_tokens(self) -> int:
        if self.last_latent_token_counts:
            return sum(self.last_latent_token_counts)
        return sum(t * v for t, v in self.latent_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D,V,C,H,W] -> [B,1,sum(T*V*H*W per layer),C,1,1]
        h_src, w_src = int(x.shape[-2]), int(x.shape[-1])
        self.last_source_hw = (h_src, w_src)
        self.last_latent_spatial_shapes = [
            (_ceil_div(h_src, s), _ceil_div(w_src, s))
            for s in self.spatial_factors
        ]
        self.last_latent_token_counts = [
            t * v * hw[0] * hw[1]
            for (t, v), hw in zip(
                self.latent_shapes, self.last_latent_spatial_shapes)
        ]
        outs = []
        for idx, block in enumerate(self.layers):
            outs.append(block(x[:, :, idx]))
        return torch.cat(outs, dim=2)


class LayeredTVExpander(nn.Module):
    """Expand a concatenated latent package back to multi-layer cylinders."""

    def __init__(self, channels: int, num_heads: int, virtual_views: int,
                 latent_shapes: list[tuple[int, int]],
                 spatial_factors: list[int]):
        super().__init__()
        self.virtual_views = int(virtual_views)
        self.latent_shapes = [(int(t), int(v)) for t, v in latent_shapes]
        self.spatial_factors = [int(s) for s in spatial_factors]
        if len(self.spatial_factors) != len(self.latent_shapes):
            raise ValueError(
                "spatial_factors must match latent_shapes length: "
                f"{len(self.spatial_factors)} vs {len(self.latent_shapes)}")
        self.layers = nn.ModuleList([
            TVJointViewExpander(channels, num_heads, t, v, virtual_views, s)
            for (t, v), s in zip(self.latent_shapes, self.spatial_factors)
        ])

    def forward(self, x: torch.Tensor,
                target_time: int,
                target_views: Optional[int] = None,
                target_hw: Optional[tuple[int, int]] = None) -> torch.Tensor:
        # x: [B,1,sum(T*V*H*W per layer),C,1,1] -> [B,T,D,V,C,H,W]
        if target_hw is None:
            raise ValueError("target_hw is required for layered TV expansion.")
        outs = []
        start = 0
        for (t_count, v_count), s, block in zip(
                self.latent_shapes, self.spatial_factors, self.layers):
            h_lat = _ceil_div(target_hw[0], s)
            w_lat = _ceil_div(target_hw[1], s)
            count = t_count * v_count * h_lat * w_lat
            part = x[:, :, start:start + count]
            outs.append(block(
                part, target_time=target_time, target_views=target_views,
                target_hw=target_hw))
            start += count
        if start != x.shape[2]:
            raise ValueError(
                "Layered latent package has unused tokens: "
                f"consumed {start}, package has {x.shape[2]}.")
        return torch.stack(outs, dim=2)


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

        # True multi-layer cylinder radii. Unlike the earlier implementation,
        # these layers stay separate through the bottleneck instead of being
        # averaged into a single RGB/feature cylinder.
        cylinder_radii = tuple(float(r) for r in getattr(
            cfg, "cylinder_radii", (4.0, 20.0)))
        if not cylinder_radii:
            raise ValueError("cylinder_radii must contain at least one radius.")
        cfg.cylinder_layer_count = len(cylinder_radii)
        base_t = int(cfg.latent_time_count)
        base_s = int(cfg.latent_spatial_downsample_factor)
        if base_s < 1:
            raise ValueError(
                "latent_spatial_downsample_factor must be >= 1, got "
                f"{base_s}.")
        if cfg.cylinder_layer_count == 1:
            latent_shapes = [(base_t, int(cfg.latent_view_count))]
            spatial_factors = [base_s]
        else:
            near_t = (
                int(cfg.near_latent_time_count)
                if cfg.near_latent_time_count is not None
                else base_t
            )
            far_t = (
                int(cfg.far_latent_time_count)
                if cfg.far_latent_time_count is not None
                else max(1, base_t // 2)
            )
            near_v = (
                int(cfg.near_latent_view_count)
                if cfg.near_latent_view_count is not None
                else int(cfg.latent_view_count)
            )
            far_v = (
                int(cfg.far_latent_view_count)
                if cfg.far_latent_view_count is not None
                else max(1, int(cfg.latent_view_count) // 2)
            )
            near_s = (
                int(cfg.near_latent_spatial_downsample_factor)
                if cfg.near_latent_spatial_downsample_factor is not None
                else base_s
            )
            far_s = (
                int(cfg.far_latent_spatial_downsample_factor)
                if cfg.far_latent_spatial_downsample_factor is not None
                else max(1, base_s * 2)
            )
            latent_shapes = [(near_t, near_v), (far_t, far_v)]
            spatial_factors = [near_s, far_s]
            if cfg.cylinder_layer_count > 2:
                latent_shapes.extend(
                    [(far_t, far_v)] * (cfg.cylinder_layer_count - 2))
                spatial_factors.extend(
                    [far_s] * (cfg.cylinder_layer_count - 2))
        if any(t < 1 or v < 1 for t, v in latent_shapes):
            raise ValueError(
                f"latent T/V counts must be positive, got {latent_shapes}")
        if any(s < 1 for s in spatial_factors):
            raise ValueError(
                "latent spatial downsample factors must be positive, got "
                f"{spatial_factors}")
        cfg.layer_latent_shapes = latent_shapes
        cfg.layer_latent_time_counts = [t for t, _v in latent_shapes]
        cfg.layer_latent_view_counts = [v for _t, v in latent_shapes]
        cfg.layer_latent_spatial_factors = spatial_factors
        cfg.layer_latent_spatial_shapes = None
        cfg.layer_latent_token_counts = [t * v for t, v in latent_shapes]
        cfg.total_latent_view_count = sum(cfg.layer_latent_token_counts)
        cfg.total_latent_token_count = cfg.total_latent_view_count
        self.virtual_projector = CylindricalViewProjector(
            cfg.virtual_view_count,
            cylinder_radii=cylinder_radii,
            ego_coordinate_mode=cfg.ego_coordinate_mode,
            cylinder_height_scale=cfg.cylinder_height_scale,
            edge_feather=cfg.projector_edge_feather,
            angular_power=cfg.projector_angular_power,
            blend_mode=cfg.projector_blend_mode,
            vertical_mode=cfg.cylinder_vertical_mode,
        )
        # Causal stem: time-causal 3D conv on the raw camera frames.
        self.stem = CausalConv3d(
            cfg.in_channels, cfg.base_channels, 3, padding=1)
        self.depth_router = CausalConv3d(
            cfg.base_channels, 1, 3, padding=1)
        nn.init.zeros_(self.depth_router.weight)
        nn.init.zeros_(self.depth_router.bias)
        self._last_depth_routing_mask: Optional[torch.Tensor] = None

        self.down1 = TimeSpaceDownBlock(
            cfg.base_channels, cfg.base_channels, stride_t=1, stride_hw=2)
        self.view_mix1 = ViewMixResBlock(cfg.base_channels)
        self.down2 = TimeSpaceDownBlock(
            cfg.base_channels, cfg.base_channels * 2, stride_t=1, stride_hw=2)
        self.view_mix2 = ViewMixResBlock(cfg.base_channels * 2)
        self.down3 = TimeSpaceDownBlock(
            cfg.base_channels * 2, cfg.base_channels * 4,
            stride_t=1,
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
        self.view_down = LayeredTVCompressor(
            cfg.base_channels * 4, cfg.num_attention_heads,
            latent_shapes, spatial_factors)
        # The latent view axis is now a concatenated near/far token package
        # with non-uniform budgets, not one uniform circular ring. Applying
        # the old circular RoPE attention here would create a fake seam
        # between far and near tokens, so the T/V joint compressors own the
        # actual bottleneck mixing.
        self.attn = nn.Identity()
        self.to_moments = nn.Linear(cfg.base_channels * 4,
                                    cfg.latent_channels * 2)

        self.from_latent = nn.Linear(cfg.latent_channels, cfg.base_channels * 4)
        self.view_up = LayeredTVExpander(
            cfg.base_channels * 4, cfg.num_attention_heads,
            cfg.virtual_view_count, latent_shapes, spatial_factors)
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
            cfg.base_channels, cfg.out_channels, 3, padding=1,
            circular_w=True)

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

    def _depth_route_mask(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B,T,V,C,H,W] -> near probability [B,T,V,1,H,W].
        b, t, v, c, height, width = h.shape
        y = h.permute(0, 2, 3, 1, 4, 5).reshape(b * v, c, t, height, width)
        mask = torch.sigmoid(self.depth_router(y))
        return mask.reshape(b, v, 1, t, height, width).permute(
            0, 3, 1, 2, 4, 5).contiguous()

    def _camera_layer_weights(self) -> Optional[torch.Tensor]:
        mask = self._last_depth_routing_mask
        if mask is None:
            return None
        d = int(getattr(self.config, "cylinder_layer_count", 1))
        if d < 2:
            return None
        near = mask.unsqueeze(2)
        far = (1.0 - mask).unsqueeze(2)
        if d == 2:
            return torch.cat([near, far], dim=2)
        far = far.expand(-1, -1, d - 1, -1, -1, -1, -1) / float(d - 1)
        return torch.cat([near, far], dim=2)

    def _project_routed_layers(
        self,
        h: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
        intrinsics_hw: Optional["tuple[int, int]"],
    ) -> torch.Tensor:
        d = int(getattr(self.config, "cylinder_layer_count", 1))
        if d < 2:
            self._last_depth_routing_mask = None
            return self.virtual_projector.project_layers(
                h, intrinsics, extrinsics, intrinsics_hw)

        mask = self._depth_route_mask(h)
        self._last_depth_routing_mask = mask
        layers = self.virtual_projector.project_layers(
            h, intrinsics, extrinsics, intrinsics_hw)
        mask_layers = self.virtual_projector.project_layers(
            mask, intrinsics, extrinsics, intrinsics_hw)
        routed = []
        routed.append(layers[:, :, 0] * mask_layers[:, :, 0])
        for layer_idx in range(1, d):
            routed.append(layers[:, :, layer_idx] *
                          (1.0 - mask_layers[:, :, layer_idx]))
        return torch.stack(routed, dim=2)

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
        cfg = self.config
        ck_enc = (
            getattr(cfg, "gradient_checkpoint_encode", False)
            and self.training
            and torch.is_grad_enabled()
        )
        if ck_enc:
            mean, logvar = checkpoint(
                lambda x_in: self._encode_trainable_body(
                    x_in, intrinsics, extrinsics, intrinsics_hw, squeezed_view,
                ),
                x,
                use_reentrant=False,
            )
        else:
            mean, logvar = self._encode_trainable_body(
                x, intrinsics, extrinsics, intrinsics_hw, squeezed_view,
            )
        output = EncoderOutput(DiagonalGaussianDistribution(mean, logvar))
        return output if return_dict else (output.latent_dist,)

    @staticmethod
    def _flatten_layers(h: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # [B,T,D,V,C,H,W] -> [B*D,T,V,C,H,W]
        b, t, d, v, c, height, width = h.shape
        h = h.permute(0, 2, 1, 3, 4, 5, 6).reshape(
            b * d, t, v, c, height, width)
        return h, b, d

    @staticmethod
    def _unflatten_layers(h: torch.Tensor, b: int, d: int) -> torch.Tensor:
        # [B*D,T,V,C,H,W] -> [B,T,D,V,C,H,W]
        _bd, t, v, c, height, width = h.shape
        return h.reshape(b, d, t, v, c, height, width)\
            .permute(0, 2, 1, 3, 4, 5, 6).contiguous()

    def _fallback_bottleneck_hw(self) -> tuple[int, int]:
        cfg = self.config
        image_h = getattr(cfg, "image_height", None)
        image_w = getattr(cfg, "image_width", None)
        if image_h is None or image_w is None:
            raise ValueError(
                "Cannot infer decoder bottleneck H/W from packed spatial "
                "latents. Decode immediately after encode, or construct the "
                "model with image_height/image_width.")
        factor = int(getattr(cfg, "spatial_downsample_factor", 8))
        return _ceil_div(int(image_h), factor), _ceil_div(int(image_w), factor)

    def _encoder_tail(self, h: torch.Tensor, squeezed_view: bool):
        """Encoder from projected cylinder layers to Gaussian moments."""
        if h.ndim == 6:
            h = h.unsqueeze(2)
        h, b, d = self._flatten_layers(h)
        h = self.down1(h)
        h = self.view_mix1(h)
        h = self.down2(h)
        h = self.view_mix2(h)
        h = self.down3(h)
        h = self.view_mix3(h)
        h = self.pre_attn(h)
        h = self._unflatten_layers(h, b, d)
        self._last_bottleneck_hw = (int(h.shape[-2]), int(h.shape[-1]))
        h = self.view_down(h)
        self.config.layer_latent_spatial_shapes = (
            self.view_down.last_latent_spatial_shapes)
        self.config.layer_latent_token_counts = (
            self.view_down.last_latent_token_counts)
        self.config.total_latent_token_count = sum(
            self.view_down.last_latent_token_counts)
        h = self.attn(h)
        moments = self.to_moments(h.permute(0, 1, 2, 4, 5, 3))
        mean, logvar = moments.chunk(2, dim=-1)
        mean = mean.permute(0, 1, 2, 5, 3, 4).contiguous()
        logvar = logvar.permute(0, 1, 2, 5, 3, 4).contiguous()
        if squeezed_view:
            mean = mean[:, :, 0].permute(0, 2, 1, 3, 4)
            logvar = logvar[:, :, 0].permute(0, 2, 1, 3, 4)
        return mean, logvar

    def _encode_trainable_body(
        self,
        x: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
        intrinsics_hw: Optional["tuple[int, int]"],
        squeezed_view: bool,
    ):
        h = self._stem(x)
        h = self._project_routed_layers(
            h, intrinsics, extrinsics, intrinsics_hw)
        return self._encoder_tail(h, squeezed_view)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        z, squeezed_view = self._as_6d(z)
        if z.shape[-2:] == (1, 1):
            target_hw = getattr(self, "_last_bottleneck_hw", None)
            if target_hw is None:
                target_hw = self._fallback_bottleneck_hw()
        else:
            target_hw = (int(z.shape[-2]), int(z.shape[-1]))
        h = self.from_latent(z.permute(0, 1, 2, 4, 5, 3))
        h = h.permute(0, 1, 2, 5, 3, 4).contiguous()
        target_time = int(getattr(self.config, "sequence_length", 5))
        h = self.view_up(
            h, target_time=target_time,
            target_views=self.config.virtual_view_count,
            target_hw=target_hw)
        h, b_layers, d_layers = self._flatten_layers(h)
        h = self.post_attn(h)
        h = self.up1(h, scale_t=1, scale_hw=2)
        h = self.view_mix_up1(h)
        h = self.up2(h, scale_t=1, scale_hw=2)
        h = self.view_mix_up2(h)
        h = self.up3(h, scale_t=1, scale_hw=2)

        v = int(h.shape[2])
        y = _to_cylinder_pano(h)
        y = self.head(y)
        y = _from_cylinder_pano(y, v)
        y = y[:, :target_time]
        y = self._unflatten_layers(y, b_layers, d_layers)
        if squeezed_view:
            y = y.mean(dim=2)[:, :, 0].permute(0, 2, 1, 3, 4)
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
        render_camera: bool = True,
    ):
        posterior = self.encode(
            x, intrinsics=intrinsics, extrinsics=extrinsics,
            intrinsics_hw=intrinsics_hw).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        cfg = self.config
        ck_dec = (
            getattr(cfg, "gradient_checkpoint_decode", False)
            and self.training
            and torch.is_grad_enabled()
            and z.requires_grad
        )
        if ck_dec:
            cylinder_reconstruction = checkpoint(
                lambda z_: self.decode(z_).sample,
                z,
                use_reentrant=False,
            )
        else:
            cylinder_reconstruction = self.decode(z).sample
        reconstruction = (
            cylinder_reconstruction.mean(dim=2)
            if cylinder_reconstruction.ndim == 7
            else cylinder_reconstruction
        )
        if (
            render_camera
            and
            getattr(cfg, "decode_camera_from_cylinder", False)
            and intrinsics is not None
            and extrinsics is not None
        ):
            x6, _ = self._as_6d(x)
            reconstruction = self.virtual_projector.render_cylinder_to_cameras(
                cylinder_reconstruction,
                intrinsics,
                extrinsics,
                intrinsics_hw,
                target_hw=(int(x6.shape[-2]), int(x6.shape[-1])),
                layer_weights=self._camera_layer_weights(),
            )
        shared_reconstruction = reconstruction
        return {
            "sample": reconstruction,
            "shared_sample": shared_reconstruction,
            "cylinder_sample": cylinder_reconstruction,
            "posterior": posterior,
            "kl_loss": posterior.kl(),
            "depth_routing_mask": self._last_depth_routing_mask,
        }
