"""Plot training metrics saved by ``train_vae.py`` -- matplotlib-free.

We generate a single SVG (vector, no dependencies, opens in any browser) with
four sub-plots:

    +-----------------+-----------------+
    | loss / rec      | kl              |
    +-----------------+-----------------+
    | learning rate   | |z| / logvar    |
    +-----------------+-----------------+

If you'd rather have a quick text summary in the terminal (no file written),
pass ``--text-only``.

Usage examples
--------------

python -m zhw_vae_510.plot_training --run zhw_vae_510/runs/nusc_scene_bs1x8_bc64
python -m zhw_vae_510.plot_training --metrics zhw_vae_510/runs/metrics.jsonl
python -m zhw_vae_510.plot_training --run zhw_vae_510/runs --text-only
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_rows(metrics_path: Path) -> list[dict]:
    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No metric rows found in {metrics_path}")
    return rows


# ---------------------------------------------------------------------------
# Tiny SVG plotting utilities -- no external deps
# ---------------------------------------------------------------------------
PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#17becf", "#7f7f7f",
]


def _nice_step(span: float, target_ticks: int = 5) -> float:
    """Pick a 'nice' tick spacing (1, 2, 5 * 10^k) covering ``span``."""
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / max(target_ticks, 1)
    exp = math.floor(math.log10(raw))
    base = raw / (10 ** exp)
    if base < 1.5:
        nice = 1
    elif base < 3.5:
        nice = 2
    elif base < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exp)


def _ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    step = _nice_step(hi - lo, target)
    start = math.floor(lo / step) * step
    out = []
    v = start
    # guard against pathological floats
    for _ in range(256):
        if v > hi + 0.5 * step:
            break
        if v >= lo - 0.5 * step:
            out.append(v)
        v += step
    return out


def _fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1000 or av < 0.01:
        return f"{v:.2g}"
    if av >= 10:
        return f"{v:.0f}"
    if av >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _series_range(series: Sequence[Sequence[float]]) -> tuple[float, float]:
    flat = [v for s in series for v in s if math.isfinite(v)]
    if not flat:
        return 0.0, 1.0
    lo, hi = min(flat), max(flat)
    if lo == hi:
        pad = abs(lo) * 0.1 or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _panel_svg(
    *,
    title: str,
    xs: Sequence[float],
    series: list[tuple[str, Sequence[float], str]],
    x_offset: float,
    y_offset: float,
    width: float,
    height: float,
    log_y: bool = False,
) -> str:
    """Return an SVG snippet for a single sub-plot."""
    pad_l, pad_r, pad_t, pad_b = 56.0, 16.0, 28.0, 36.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    if not xs:
        return ""

    x_lo, x_hi = min(xs), max(xs)
    if x_lo == x_hi:
        x_hi = x_lo + 1.0

    raw_values = [s[1] for s in series]
    if log_y:
        # Drop non-positive values for log scaling.
        clean = []
        for s in raw_values:
            clean.append([v for v in s if v > 0 and math.isfinite(v)])
        if not any(clean):
            log_y = False
            y_lo, y_hi = _series_range(raw_values)
        else:
            y_lo_lin = min(min(s) for s in clean if s)
            y_hi_lin = max(max(s) for s in clean if s)
            y_lo = math.log10(y_lo_lin)
            y_hi = math.log10(y_hi_lin)
            if y_lo == y_hi:
                y_lo -= 0.5
                y_hi += 0.5
            else:
                pad = (y_hi - y_lo) * 0.05
                y_lo -= pad
                y_hi += pad
    else:
        y_lo, y_hi = _series_range(raw_values)

    def sx(v: float) -> float:
        return pad_l + (v - x_lo) / (x_hi - x_lo) * plot_w

    def sy(v: float) -> float:
        if log_y:
            if v <= 0 or not math.isfinite(v):
                return pad_t + plot_h  # clamp to bottom
            v = math.log10(v)
        # Invert: data low -> bottom of plot.
        return pad_t + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h

    parts: list[str] = []
    parts.append(f'<g transform="translate({x_offset:.2f},{y_offset:.2f})">')

    # Background + frame.
    parts.append(
        f'<rect x="{pad_l:.2f}" y="{pad_t:.2f}" width="{plot_w:.2f}" '
        f'height="{plot_h:.2f}" fill="#ffffff" stroke="#888" stroke-width="1"/>'
    )

    # Title.
    parts.append(
        f'<text x="{pad_l + plot_w/2:.2f}" y="{pad_t - 10:.2f}" '
        f'text-anchor="middle" font-family="Arial,Helvetica,sans-serif" '
        f'font-size="13" fill="#222">{html.escape(title)}</text>'
    )

    # Y ticks + grid.
    y_tick_vals = (
        list(range(int(math.floor(y_lo)), int(math.ceil(y_hi)) + 1))
        if log_y else _ticks(y_lo, y_hi, 5)
    )
    for t in y_tick_vals:
        if t < y_lo or t > y_hi:
            continue
        # In log mode the math: sy expects a *linear* value, but our y_lo/y_hi
        # are already in log space. Build a tiny shim:
        if log_y:
            yp = pad_t + plot_h - (t - y_lo) / (y_hi - y_lo) * plot_h
            label = f"1e{int(t)}" if t == int(t) else f"1e{t:.1f}"
        else:
            yp = sy(t)
            label = _fmt_tick(t)
        parts.append(
            f'<line x1="{pad_l:.2f}" y1="{yp:.2f}" x2="{pad_l + plot_w:.2f}" '
            f'y2="{yp:.2f}" stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6:.2f}" y="{yp + 3:.2f}" text-anchor="end" '
            f'font-family="Arial,Helvetica,sans-serif" font-size="10" '
            f'fill="#444">{html.escape(label)}</text>'
        )

    # X ticks.
    for t in _ticks(x_lo, x_hi, 5):
        if t < x_lo or t > x_hi:
            continue
        xp = sx(t)
        parts.append(
            f'<line x1="{xp:.2f}" y1="{pad_t:.2f}" x2="{xp:.2f}" '
            f'y2="{pad_t + plot_h:.2f}" stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xp:.2f}" y="{pad_t + plot_h + 14:.2f}" '
            f'text-anchor="middle" font-family="Arial,Helvetica,sans-serif" '
            f'font-size="10" fill="#444">{html.escape(_fmt_tick(t))}</text>'
        )

    # X label.
    parts.append(
        f'<text x="{pad_l + plot_w/2:.2f}" y="{pad_t + plot_h + 28:.2f}" '
        f'text-anchor="middle" font-family="Arial,Helvetica,sans-serif" '
        f'font-size="11" fill="#222">step</text>'
    )

    # Series polylines.
    for label, ys, color in series:
        pts = []
        for x, y in zip(xs, ys):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if log_y and y <= 0:
                continue
            pts.append(f"{sx(x):.2f},{sy(y):.2f}")
        if not pts:
            continue
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.6" '
            f'points="{" ".join(pts)}"/>'
        )

    # Legend.
    legend_y = pad_t + 6
    legend_x = pad_l + 8
    for i, (label, _ys, color) in enumerate(series):
        ly = legend_y + i * 14
        parts.append(
            f'<rect x="{legend_x:.2f}" y="{ly - 8:.2f}" width="14" height="3" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 18:.2f}" y="{ly - 2:.2f}" '
            f'font-family="Arial,Helvetica,sans-serif" font-size="11" '
            f'fill="#222">{html.escape(label)}</text>'
        )

    parts.append("</g>")
    return "\n".join(parts)


def render_svg(rows: list[dict], run_name: str) -> str:
    steps = [r["step"] for r in rows]

    panels = [
        # (title, [(label, key, color), ...], log_y)
        ("Loss",          [("loss", "loss", PALETTE[0]),
                           ("rec",  "rec",  PALETTE[1])], False),
        ("KL",            [("kl",   "kl",   PALETTE[2])], False),
        ("Learning Rate", [("lr",   "lr",   PALETTE[3])], True),
        ("Latent Stats",  [("|z|",         "mean_abs_z",  PALETTE[0]),
                           ("logvar_mean", "logvar_mean", PALETTE[1])], False),
    ]

    panel_w, panel_h = 520, 320
    title_h = 36
    cols = 2
    rows_grid = 2
    total_w = panel_w * cols
    total_h = title_h + panel_h * rows_grid

    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" '
        f'height="{total_h}" viewBox="0 0 {total_w} {total_h}">'
    )
    out.append('<rect width="100%" height="100%" fill="#fafafa"/>')
    out.append(
        f'<text x="{total_w/2:.2f}" y="22" text-anchor="middle" '
        f'font-family="Arial,Helvetica,sans-serif" font-size="16" '
        f'fill="#111">{html.escape("Training Curves: " + run_name)}</text>'
    )

    for i, (title, spec, log_y) in enumerate(panels):
        col = i % cols
        row = i // cols
        x_off = col * panel_w
        y_off = title_h + row * panel_h
        series = []
        for label, key, color in spec:
            ys = [float(r.get(key, math.nan)) for r in rows]
            series.append((label, ys, color))
        out.append(_panel_svg(
            title=title, xs=steps, series=series,
            x_offset=x_off, y_offset=y_off,
            width=panel_w, height=panel_h, log_y=log_y,
        ))

    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Text summary fallback
# ---------------------------------------------------------------------------
def _last_n_avg(values: Iterable[float], n: int = 50) -> float:
    vs = [v for v in values if math.isfinite(v)]
    if not vs:
        return float("nan")
    tail = vs[-n:]
    return sum(tail) / len(tail)


def print_text_summary(rows: list[dict]):
    keys = ["loss", "rec", "kl", "kl_weight_now",
            "mean_abs_z", "logvar_mean", "lr", "it_s"]
    last = rows[-1]
    print(f"  rows: {len(rows)}, last step: {last['step']}")
    print(f"  {'metric':<16} {'last':>12} {'avg(last 50)':>14}")
    for k in keys:
        if k not in last:
            continue
        avg = _last_n_avg([r.get(k, math.nan) for r in rows], 50)
        print(f"  {k:<16} {last[k]!s:>12} {avg:>14.4g}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="",
                    help="Run directory containing metrics.jsonl.")
    ap.add_argument("--metrics", default="",
                    help="Path to metrics.jsonl. Overrides --run.")
    ap.add_argument("--out", default="",
                    help="Output SVG path. Defaults to <run>/training_curves.svg.")
    ap.add_argument("--text-only", action="store_true",
                    help="Skip SVG generation, just print a text summary.")
    args = ap.parse_args()

    if args.metrics:
        metrics_path = Path(args.metrics)
        run_dir = metrics_path.parent
    else:
        run_dir = Path(args.run or "zhw_vae_510/runs")
        metrics_path = run_dir / "metrics.jsonl"

    rows = load_rows(metrics_path)
    print(f"[info] metrics source: {metrics_path}")
    print(f"[info] points: {len(rows)}, last step: {rows[-1]['step']}")

    if args.text_only:
        print_text_summary(rows)
        return

    out_path = Path(args.out) if args.out else run_dir / "training_curves.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    svg = render_svg(rows, run_name=run_dir.name)
    out_path.write_text(svg, encoding="utf-8")
    print(f"[done] wrote {out_path}")
    print("[hint] open the .svg in any browser, or pass --text-only for a CLI summary.")


if __name__ == "__main__":
    main()
