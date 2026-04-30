"""Visualize SurgWMBench predictions vs ground-truth sparse anchors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .data import SurgWMBenchClipDataset, collate_sparse_anchors
from .eval_surgwmbench import load_model, predict_sparse
from .train_surgwmbench import _data_root
from .utils import get_device, move_to_device, seed_everything


GT_COLOR = (0, 220, 60)        # green
PRED_COLOR = (240, 60, 60)     # red
GT_LINE = (0, 180, 50)
PRED_LINE = (220, 40, 40)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _frame_for_anchor(frame_paths: list[str], anchor_idx: int, choice: str) -> Path:
    n = len(frame_paths)
    if choice == "first":
        return Path(frame_paths[0])
    if choice == "last":
        return Path(frame_paths[n - 1])
    if choice == "middle":
        return Path(frame_paths[n // 2])
    return Path(frame_paths[anchor_idx])


def _draw_trajectory(
    base: Image.Image,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    radius: int,
    label_offset: tuple[int, int],
    font: ImageFont.ImageFont,
    show_indices: bool,
) -> None:
    draw = ImageDraw.Draw(base, mode="RGBA")
    if len(points) >= 2:
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            draw.line([(x0, y0), (x1, y1)], fill=line_color + (220,), width=3)
    for i, (x, y) in enumerate(points):
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color + (220,),
            outline=(0, 0, 0, 255),
            width=2,
        )
        if show_indices and (i == 0 or i == len(points) - 1 or i % 5 == 0):
            draw.text((x + label_offset[0], y + label_offset[1]), str(i), font=font, fill=color + (255,))


def _legend(image: Image.Image, font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(image, mode="RGBA")
    box = (12, 12, 240, 90)
    draw.rectangle(box, fill=(0, 0, 0, 160), outline=(255, 255, 255, 200), width=2)
    draw.ellipse((28, 26, 48, 46), fill=GT_COLOR + (255,), outline=(0, 0, 0, 255), width=2)
    draw.text((58, 24), "Ground Truth", font=font, fill=(255, 255, 255, 255))
    draw.ellipse((28, 56, 48, 76), fill=PRED_COLOR + (255,), outline=(0, 0, 0, 255), width=2)
    draw.text((58, 54), "Prediction", font=font, fill=(255, 255, 255, 255))


def render_clip(
    item: dict,
    pred_norm: torch.Tensor,
    output_path: Path,
    frame_choice: str,
    show_indices: bool,
    radius: int,
) -> None:
    image_h, image_w = (int(item["image_size_original"][0]), int(item["image_size_original"][1]))
    gt_px = item["human_anchor_coords_px"].cpu().numpy()
    pred_px = pred_norm.cpu().numpy() * [image_w, image_h]

    base_path = _frame_for_anchor(item["frame_paths"], 0, frame_choice)
    with Image.open(base_path) as raw:
        canvas = raw.convert("RGBA")

    font = _font(max(14, image_h // 60))
    _draw_trajectory(
        canvas,
        [(float(x), float(y)) for x, y in gt_px],
        color=GT_COLOR, line_color=GT_LINE,
        radius=radius, label_offset=(radius + 4, -radius - 4),
        font=font, show_indices=show_indices,
    )
    _draw_trajectory(
        canvas,
        [(float(x), float(y)) for x, y in pred_px],
        color=PRED_COLOR, line_color=PRED_LINE,
        radius=radius, label_offset=(radius + 4, radius + 2),
        font=font, show_indices=show_indices,
    )
    _legend(canvas, font)

    title_font = _font(max(18, image_h // 50))
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    err_px = ((gt_px - pred_px) ** 2).sum(axis=-1) ** 0.5
    title = (
        f"trajectory_id={item['trajectory_id']}  "
        f"difficulty={item.get('difficulty', '?')}  "
        f"frame={base_path.name}  "
        f"pixel_ADE={err_px.mean():.1f}  pixel_FDE={err_px[-1]:.1f}"
    )
    draw.rectangle((0, image_h - 48, image_w, image_h), fill=(0, 0, 0, 180))
    draw.text((16, image_h - 38), title, font=title_font, fill=(255, 255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, format="PNG", optimize=True)


@torch.inference_mode()
def visualize(args: argparse.Namespace) -> None:
    device = get_device()
    model, config = load_model(args.checkpoint, args.config, device)
    seed_everything(config.train.seed)
    if args.interpolation_method:
        config.data.interpolation_method = args.interpolation_method

    dataset = SurgWMBenchClipDataset(
        _data_root(args, config),
        args.manifest or config.data.test_manifest,
        interpolation_method=config.data.interpolation_method,
        image_size=config.model.image_size,
        frame_sampling="sparse_anchors",
    )
    n = len(dataset) if args.num_clips is None else min(args.num_clips, len(dataset))
    output_dir = Path(args.output_dir).expanduser()

    model.eval()
    print(f"rendering {n}/{len(dataset)} clips to {output_dir}")
    for idx in range(n):
        item = dataset[idx]
        batch = collate_sparse_anchors([item])
        batch = move_to_device(batch, device)
        pred, _, _ = predict_sparse(model, batch, use_time_delta=config.model.use_time_delta)
        out_name = f"{idx:04d}_{item['source_video_id']}_{item['trajectory_id']}.png"
        render_clip(
            item,
            pred[0],
            output_dir / out_name,
            args.frame,
            show_indices=args.show_indices,
            radius=args.radius,
        )
        if (idx + 1) % 10 == 0 or idx == n - 1:
            print(f"  [{idx + 1}/{n}] saved {out_name}")
    print(f"done. images at: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", "--data-root", dest="dataset_root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--config", default="configs/surgwmbench_mwm_1gpu.yaml")
    parser.add_argument("--interpolation-method", default=None)
    parser.add_argument("--num-clips", type=int, default=None,
                        help="Limit number of clips to visualize (default: all).")
    parser.add_argument("--frame", choices=["first", "middle", "last"], default="middle",
                        help="Which clip frame to use as background canvas.")
    parser.add_argument("--radius", type=int, default=10, help="Anchor circle radius in pixels.")
    parser.add_argument("--show-indices", action="store_true",
                        help="Annotate anchor indices on the image.")
    return parser.parse_args()


def main() -> None:
    visualize(parse_args())


if __name__ == "__main__":
    main()
