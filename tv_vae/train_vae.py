"""Train the camera-domain TV VAE.

This script is intentionally smaller than the old cylinder pipeline.  It is
for ablations on the bottleneck:

- joint: per-spatial-column cross-attention over time and scene tokens.
- non_joint: separable time/scene pooling with the same token budget, without
  joint attention.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

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
    from .losses import VAELoss, VAELossConfig  # noqa: E402
    from .model import TVVAE  # noqa: E402
except ImportError:  # pragma: no cover
    from data import (  # noqa: E402
        MultiViewVAEAdapter,
        SyntheticConfig,
        SyntheticCylinderDataset,
        make_nuscenes_base,
        make_nuscenes_scene_folder_base,
        vae_collate,
    )
    from losses import VAELoss, VAELossConfig  # noqa: E402
    from model import TVVAE  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "nuplan", "nus", "nuscenes", "nuscenes_scene"],
                   default="synthetic")
    p.add_argument("--out", default="tv_vae/runs")
    p.add_argument("--resume", default="")

    p.add_argument("--image-hw", type=int, nargs=2, default=[64, 128])
    p.add_argument("--sequence-length", type=int, default=5)
    p.add_argument("--view-count", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--overfit-samples", type=int, default=0)
    p.add_argument("--overfit-sample-index", type=int, default=0)

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

    p.add_argument("--rec-kind", default="l1", choices=["l1", "l2", "huber"])
    p.add_argument("--kl-weight", type=float, default=1e-6)
    p.add_argument("--kl-reduction", choices=["batch", "mean"], default="mean")
    p.add_argument("--kl-warmup", type=int, default=2000)
    p.add_argument("--perceptual-weight", type=float, default=0.0)
    p.add_argument("--perceptual-batch-size", type=int, default=4)
    p.add_argument("--edge-loss-weight", type=float, default=0.0)
    p.add_argument("--logvar-reg-weight", type=float, default=0.0)
    p.add_argument("--sample-posterior", action="store_true")

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)

    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    p.add_argument("--ddp-backend", default="")
    p.add_argument("--find-unused-parameters", action="store_true")
    p.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int,
                   default=int(os.environ.get("LOCAL_RANK", "0")),
                   help=argparse.SUPPRESS)

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--preview-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=1000)
    return p


def device_of() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not distributed:
        return {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device_of(),
        }

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = args.ddp_backend or "nccl"
    else:
        device = torch.device("cpu")
        backend = args.ddp_backend or "gloo"
    dist.init_process_group(backend=backend)
    return {
        "distributed": True,
        "rank": dist.get_rank(),
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def is_main_process(ddp_info) -> bool:
    return int(ddp_info.get("rank", 0)) == 0


def rank0_print(ddp_info, *args, **kwargs):
    if is_main_process(ddp_info):
        print(*args, **kwargs)


def cleanup_distributed(ddp_info):
    if ddp_info.get("distributed") and dist.is_initialized():
        dist.destroy_process_group()


def amp_dtype(name: str):
    return {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def lr_lambda(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    if step < warmup:
        return step / float(warmup)
    return 1.0


def make_dataset(args, ddp_info=None):
    if args.data == "synthetic":
        cfg = SyntheticConfig(
            samples_per_epoch=10_000,
            sequence_length=args.sequence_length,
            view_count=args.view_count,
            image_hw=tuple(args.image_hw),
        )
        return SyntheticCylinderDataset(cfg)

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
        if ddp_info is None or is_main_process(ddp_info):
            print(f"[data] nuScenes: {len(base)} clips "
                  f"(dataset_name={args.nusc_dataset_name}, split={args.nusc_split}, "
                  f"T={args.sequence_length})")
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
        if ddp_info is None or is_main_process(ddp_info):
            print(f"[data] nuScenes scene-folder: {len(base)} clips "
                  f"(root={args.nusc_data_root}, split={args.nusc_split}, "
                  f"T={args.sequence_length})")
        return MultiViewVAEAdapter(
            base, sequence_length=args.sequence_length,
            image_hw=tuple(args.image_hw))

    from dwm.tools.dataset_nus import make_base_ds
    base = make_base_ds(train=True)
    return MultiViewVAEAdapter(
        base, sequence_length=args.sequence_length,
        image_hw=tuple(args.image_hw))


def maybe_make_overfit_subset(dataset, args, ddp_info):
    if args.overfit_samples <= 0:
        return dataset
    end = args.overfit_sample_index + args.overfit_samples
    if args.overfit_sample_index < 0 or end > len(dataset):
        raise ValueError(
            "Requested overfit subset is outside dataset: "
            f"start={args.overfit_sample_index}, "
            f"count={args.overfit_samples}, len={len(dataset)}")
    indices = list(range(args.overfit_sample_index, end))
    rank0_print(
        ddp_info,
        f"[data] overfit subset enabled: indices {indices[0]}..{indices[-1]} "
        f"({len(indices)} sample(s)); shuffle disabled")
    return Subset(dataset, indices)


def save_preview(images: torch.Tensor, recon: torch.Tensor, path: str,
                 max_views: int = 6):
    b, t, v, c, h, w = images.shape
    v_keep = min(v, max_views)
    gt = images[0, 0, :v_keep].detach().cpu()
    rc = recon[0, 0, :v_keep].detach().cpu()
    grid = torch.cat([gt, rc], dim=0)
    grid = (grid.clamp(-1, 1) + 1) / 2.0
    vutils.save_image(grid, path, nrow=v_keep)


def log_input_probe(ddp_info, sample: dict, out_dir: Path, max_views: int = 6):
    if not is_main_process(ddp_info):
        return
    imgs = sample["vae_images"]
    view_means = imgs.mean(dim=(0, 2, 3, 4))
    means_str = ", ".join(f"{m:.3f}" for m in view_means.tolist())
    print(
        f"[data] probe vae_images={tuple(imgs.shape)} "
        f"range=[{imgs.min().item():.3f}, {imgs.max().item():.3f}] "
        f"view_mean=[{means_str}]")
    v_keep = min(imgs.shape[1], max_views)
    grid = (imgs[0, :v_keep].clamp(-1, 1) + 1) / 2.0
    vutils.save_image(grid, str(out_dir / "input_probe.png"), nrow=v_keep)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor,
                        kind: str) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "l2":
        return F.mse_loss(pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(pred, target, beta=0.1)
    raise ValueError(f"Unknown rec_kind={kind}")


def gradient_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor):
    dx_loss = (pred[..., :, 1:] - pred[..., :, :-1] -
               (target[..., :, 1:] - target[..., :, :-1])).abs()
    dy_loss = (pred[..., 1:, :] - pred[..., :-1, :] -
               (target[..., 1:, :] - target[..., :-1, :])).abs()
    return 0.5 * (dx_loss.mean() + dy_loss.mean())


def reduce_metric_dict(metrics: dict, ddp_info) -> dict:
    reduced = {}
    for key, value in metrics.items():
        if not torch.is_tensor(value):
            value = torch.as_tensor(float(value), device=ddp_info["device"])
        else:
            value = value.detach()
        if ddp_info["distributed"]:
            value = value.clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= ddp_info["world_size"]
        reduced[key] = value
    return reduced


def save_checkpoint(model, optim, scheduler, step, args, path):
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "args": vars(args),
    }
    torch.save(payload, path)


def load_model_state(model, state_dict, strict: bool = False):
    try:
        return model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if all(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            return model.load_state_dict(state_dict, strict=strict)
        raise


def append_metrics(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    args = build_parser().parse_args()
    ddp_info = init_distributed(args)
    try:
        device = ddp_info["device"]
        out_dir = Path(args.out)
        preview_dir = out_dir / "previews"
        ckpt_dir = out_dir / "ckpts"
        metrics_path = out_dir / "metrics.jsonl"
        if is_main_process(ddp_info):
            out_dir.mkdir(parents=True, exist_ok=True)
            preview_dir.mkdir(exist_ok=True)
            ckpt_dir.mkdir(exist_ok=True)
            if not args.resume and metrics_path.exists():
                metrics_path.unlink()
        if ddp_info["distributed"]:
            if device.type == "cuda":
                dist.barrier(device_ids=[ddp_info["local_rank"]])
            else:
                dist.barrier()

        rank0_print(
            ddp_info,
            f"[setup] device={device}, amp={args.amp}, "
            f"distributed={ddp_info['distributed']}, "
            f"world_size={ddp_info['world_size']}, "
            f"posterior={'sample' if args.sample_posterior else 'mode'}")

        dataset = make_dataset(args, ddp_info)
        dataset = maybe_make_overfit_subset(dataset, args, ddp_info)
        overfit_active = args.overfit_samples > 0
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_info["world_size"],
            rank=ddp_info["rank"],
            shuffle=not overfit_active,
            drop_last=not overfit_active,
        ) if ddp_info["distributed"] else None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None and not overfit_active),
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=vae_collate,
            drop_last=not overfit_active,
            pin_memory=device.type == "cuda")

        probe = dataset[0]
        log_input_probe(ddp_info, probe, out_dir)
        v_input = int(probe["vae_images"].shape[1])
        args.view_count = v_input
        if args.latent_scene_token_count > args.scene_token_count:
            raise ValueError(
                "latent_scene_token_count must be <= scene_token_count.")
        if args.latent_time_count > args.sequence_length:
            raise ValueError(
                f"latent_time_count={args.latent_time_count} > T={args.sequence_length}")

        if is_main_process(ddp_info):
            with open(out_dir / "args.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)

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

        n_params = sum(p.numel() for p in model.parameters())
        h_lat = (args.image_hw[0] + args.spatial_downsample_factor - 1) // args.spatial_downsample_factor
        w_lat = (args.image_hw[1] + args.spatial_downsample_factor - 1) // args.spatial_downsample_factor
        latent_tv_tokens = (
            args.latent_time_count * args.latent_scene_token_count)
        latent_tokens = latent_tv_tokens * h_lat * w_lat
        source_tokens = args.sequence_length * v_input * args.image_hw[0] * args.image_hw[1]
        rank0_print(
            ddp_info,
            f"[model] TVVAE/{args.tv_compression}: {n_params/1e6:.2f}M params, "
            f"scene_tokens={args.scene_token_count}, "
            f"latent_TS_tokens={latent_tv_tokens}, "
            f"latent_tokens={latent_tokens}, "
            f"raw_TVHW_tokens={source_tokens}, "
            f"compression={source_tokens / max(latent_tokens, 1):.1f}x")

        loss_fn = VAELoss(VAELossConfig(
            rec_kind=args.rec_kind,
            kl_weight=args.kl_weight,
            kl_reduction=args.kl_reduction,
            kl_warmup_steps=args.kl_warmup,
            perceptual_weight=args.perceptual_weight,
            perceptual_batch_size=args.perceptual_batch_size,
            logvar_reg_weight=args.logvar_reg_weight,
        )).to(device)
        optim = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: lr_lambda(s, args.warmup_steps))

        start_step = 0
        if args.resume and os.path.exists(args.resume):
            rank0_print(ddp_info, f"[resume] loading {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu")
            load_model_state(model, ckpt["model"], strict=False)
            if ckpt.get("optimizer") is not None:
                optim.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_step = int(ckpt.get("step", 0))
            loss_fn._step = start_step

        if ddp_info["distributed"]:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[ddp_info["local_rank"]] if device.type == "cuda" else None,
                output_device=ddp_info["local_rank"] if device.type == "cuda" else None,
                find_unused_parameters=args.find_unused_parameters)

        autocast_dtype = amp_dtype(args.amp)
        use_amp = autocast_dtype is not None
        scaler = torch.cuda.amp.GradScaler() if (
            use_amp and autocast_dtype == torch.float16 and device.type == "cuda"
        ) else None

        model.train()
        step = start_step
        epoch = start_step // max(len(loader), 1)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterator = iter(loader)
        log_buf = {}
        t_last = time.time()

        while step < args.steps:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)

            x = batch["vae_images"].to(device, non_blocking=True)
            K = batch.get("camera_intrinsics")
            E = batch.get("camera_transforms")
            intr_hw = batch.get("intrinsics_hw")
            if K is not None and E is not None:
                K = K.to(device, non_blocking=True)
                E = E.to(device, non_blocking=True)
            else:
                K = None
                E = None
            optim.zero_grad(set_to_none=True)
            ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if use_amp else torch.enable_grad()
            )
            with ctx:
                output = model(
                    x,
                    sample_posterior=args.sample_posterior,
                    intrinsics=K,
                    extrinsics=E,
                    intrinsics_hw=intr_hw,
                )
                recon = output["sample"]
                posterior = output["posterior"]
                if recon.shape != x.shape:
                    raise RuntimeError(
                        f"recon shape {tuple(recon.shape)} != input {tuple(x.shape)}")
                rec = reconstruction_loss(recon, x, args.rec_kind)
                perceptual = loss_fn._perceptual(recon, x)
                edge = (
                    gradient_reconstruction_loss(recon, x)
                    if args.edge_loss_weight > 0 else recon.new_zeros(()))
                kl = loss_fn._kl(posterior)
                kl_w = loss_fn._kl_weight()
                logvar_reg = (
                    posterior.logvar.pow(2).mean()
                    if args.logvar_reg_weight > 0 else recon.new_zeros(()))
                loss = (
                    rec
                    + args.perceptual_weight * perceptual
                    + args.edge_loss_weight * edge
                    + kl_w * kl
                    + args.logvar_reg_weight * logvar_reg
                )
                loss_fn._step += 1
                metrics = {
                    "loss": loss,
                    "rec": rec.detach(),
                    "perceptual": perceptual.detach(),
                    "edge": edge.detach(),
                    "kl": kl.detach(),
                    "kl_weight_now": recon.new_tensor(kl_w),
                    "logvar_reg": logvar_reg.detach(),
                    "mean_abs_z": posterior.mean.detach().abs().mean(),
                    "logvar_mean": posterior.logvar.detach().mean(),
                }

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
            scheduler.step()
            step += 1

            reduced = reduce_metric_dict(metrics, ddp_info)
            for key, value in reduced.items():
                log_buf.setdefault(key, []).append(float(value.item()))

            if step % args.log_every == 0:
                if is_main_process(ddp_info):
                    avg = {k: sum(vs) / len(vs) for k, vs in log_buf.items()}
                    dt = time.time() - t_last
                    t_last = time.time()
                    lr_now = optim.param_groups[0]["lr"]
                    it_s = args.log_every / max(dt, 1e-6)
                    append_metrics(metrics_path, {
                        "step": step,
                        "steps_total": args.steps,
                        "loss": avg["loss"],
                        "rec": avg["rec"],
                        "perceptual": avg.get("perceptual", 0.0),
                        "edge": avg.get("edge", 0.0),
                        "kl": avg["kl"],
                        "kl_weight_now": avg["kl_weight_now"],
                        "weighted_kl": avg["kl"] * avg["kl_weight_now"],
                        "logvar_reg": avg.get("logvar_reg", 0.0),
                        "mean_abs_z": avg["mean_abs_z"],
                        "logvar_mean": avg["logvar_mean"],
                        "lr": lr_now,
                        "it_s": it_s,
                    })
                    print(
                        f"[step {step:6d}/{args.steps}] "
                        f"loss={avg['loss']:.4f} rec={avg['rec']:.4f} "
                        f"edge={avg.get('edge', 0.0):.4f} "
                        f"perc={avg.get('perceptual', 0.0):.4f} "
                        f"kl={avg['kl']:.4f} "
                        f"(w={avg['kl_weight_now']:.1e}, "
                        f"wkl={avg['kl'] * avg['kl_weight_now']:.4f}) "
                        f"|z|={avg['mean_abs_z']:.3f} "
                        f"logvar~{avg['logvar_mean']:.2f} "
                        f"lr={lr_now:.2e} ({it_s:.1f} it/s)")
                log_buf.clear()

            if is_main_process(ddp_info) and step % args.preview_every == 0:
                with torch.no_grad():
                    model.eval()
                    raw_model = model.module if hasattr(model, "module") else model
                    preview = raw_model(
                        x,
                        sample_posterior=False,
                        intrinsics=K,
                        extrinsics=E,
                        intrinsics_hw=intr_hw,
                    )["sample"]
                    save_preview(
                        x.detach(), preview.detach(),
                        str(preview_dir / f"step_{step:06d}.png"))
                    model.train()

            if is_main_process(ddp_info) and (
                step % args.ckpt_every == 0 or step == args.steps
            ):
                save_checkpoint(
                    model, optim, scheduler, step, args,
                    str(ckpt_dir / f"step_{step:06d}.pt"))
                save_checkpoint(
                    model, optim, scheduler, step, args,
                    str(ckpt_dir / "last.pt"))

        rank0_print(ddp_info, "[done] training finished")
    finally:
        cleanup_distributed(ddp_info)


if __name__ == "__main__":
    main()
