from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Any

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", default="gpu")
_pre.add_argument("--gpu", default="0")
_pre.add_argument("--project_root", default="")
_pre_args, _ = _pre.parse_known_args()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if _pre_args.device.lower() == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _pre_args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

if _pre_args.project_root:
    sys.path.insert(0, str(Path(_pre_args.project_root).resolve()))

import cv2
import numpy as np
import tensorflow as tf

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x=None, **kwargs):
        return x if x is not None else range(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LimFUNet aggregation diagnostics for images or videos.")
    p.add_argument("--mode", choices=["images", "videos"], required=True)
    p.add_argument("--image", default="")
    p.add_argument("--video", default="")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--model", action="append", default=[], help="Use G=/path/to/model.h5. Repeat per model.")
    p.add_argument("--load_mode", choices=["auto", "full", "weights"], default="auto")

    p.add_argument("--input_height", type=int, default=416)
    p.add_argument("--input_width", type=int, default=608)
    p.add_argument("--n_classes", type=int, default=2)
    p.add_argument("--fire_class", type=int, default=1)
    p.add_argument("--vote_threshold", type=float, default=0.5)

    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--gpu", default="0")
    p.add_argument("--project_root", default="")

    p.add_argument("--colormap", choices=["JET", "TURBO", "HOT", "INFERNO", "VIRIDIS"], default="JET")
    p.add_argument("--overlay", action="store_true")
    p.add_argument("--overlay_alpha", type=float, default=0.45)
    p.add_argument("--save_model_masks", action="store_true")
    p.add_argument("--save_video_frames", action="store_true")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--codec", default="mp4v")
    return p.parse_args()


def configure_tf() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def parse_model_specs(specs: List[str]) -> List[Tuple[int, Path]]:
    out = []
    for spec in specs:
        if "=" in spec:
            g, p = spec.split("=", 1)
        elif ":" in spec:
            g, p = spec.split(":", 1)
        else:
            raise ValueError(f"Bad --model spec: {spec}. Use G=/path/to/model.h5")

        path = Path(p.strip().strip('"').strip("'")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file for G={g}: {path}")
        out.append((int(g.strip()), path))

    if not out:
        raise ValueError("Provide at least one --model G=/path/to/model.h5")

    return sorted(out, key=lambda x: x[0])


def import_limfunet_builder():
    try:
        from keras_segmentation.models.unet import limfunet
        return limfunet
    except Exception as e:
        raise ImportError(
            "Could not import keras_segmentation.models.unet.limfunet. "
            "Use --load_mode full for full saved .h5 models, or run from the LimFUNet project root."
        ) from e


def infer_input_hw(model: tf.keras.Model, fallback_h: int, fallback_w: int) -> Tuple[int, int, str]:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]

    if len(shape) == 4 and shape[-1] == 3:
        h = int(shape[1]) if shape[1] is not None else fallback_h
        w = int(shape[2]) if shape[2] is not None else fallback_w
        return h, w, "channels_last"

    if len(shape) == 4 and shape[1] == 3:
        h = int(shape[2]) if shape[2] is not None else fallback_h
        w = int(shape[3]) if shape[3] is not None else fallback_w
        return h, w, "channels_first"

    return fallback_h, fallback_w, "channels_last"


def infer_n_classes(model: tf.keras.Model, fallback: int) -> int:
    shape = model.output_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) >= 3 and shape[-1] is not None:
        return int(shape[-1])
    return fallback


def factor_hw(n: int, ref_h: int, ref_w: int) -> Tuple[int, int]:
    if n == ref_h * ref_w:
        return ref_h, ref_w

    target_ratio = ref_w / max(ref_h, 1)
    best_h, best_w, best_err = ref_h, ref_w, float("inf")
    root = int(np.sqrt(n)) + 2

    for h in range(1, root + 1):
        if n % h == 0:
            w = n // h
            err = abs((w / h) - target_ratio)
            if err < best_err:
                best_h, best_w, best_err = h, w, err

    return int(best_h), int(best_w)


def infer_output_hw(model: tf.keras.Model, input_h: int, input_w: int, n_classes: int) -> Tuple[int, int]:
    if hasattr(model, "output_height") and hasattr(model, "output_width"):
        return int(model.output_height), int(model.output_width)

    shape = model.output_shape
    if isinstance(shape, list):
        shape = shape[0]

    if len(shape) == 4:
        if shape[-1] == n_classes:
            return int(shape[1]), int(shape[2])
        if shape[1] == n_classes:
            return int(shape[2]), int(shape[3])

    if len(shape) == 3:
        tokens = int(shape[1])
        return factor_hw(tokens, input_h, input_w)

    return input_h, input_w


def load_one_model(g: int, path: Path, args: argparse.Namespace):
    if args.load_mode in {"auto", "full"}:
        try:
            model = tf.keras.models.load_model(str(path), compile=False)
            ih, iw, ordering = infer_input_hw(model, args.input_height, args.input_width)
            nc = infer_n_classes(model, args.n_classes)
            oh, ow = infer_output_hw(model, ih, iw, nc)
            print(f"Loaded full model G={g}: {path}")
            return {"G": g, "model": model, "input_h": ih, "input_w": iw, "output_h": oh, "output_w": ow, "n_classes": nc, "ordering": ordering}
        except Exception as e:
            if args.load_mode == "full":
                raise RuntimeError(f"Failed to load full model for G={g}: {path}\n{e}") from e

    limfunet = import_limfunet_builder()
    model = limfunet(n_classes=args.n_classes, input_height=args.input_height, input_width=args.input_width, G=g)
    model.load_weights(str(path))
    print(f"Loaded weights G={g}: {path}")

    return {
        "G": g,
        "model": model,
        "input_h": int(getattr(model, "input_height", args.input_height)),
        "input_w": int(getattr(model, "input_width", args.input_width)),
        "output_h": int(getattr(model, "output_height", args.input_height)),
        "output_w": int(getattr(model, "output_width", args.input_width)),
        "n_classes": int(getattr(model, "n_classes", args.n_classes)),
        "ordering": "channels_last",
    }


def load_models(specs: List[Tuple[int, Path]], args: argparse.Namespace):
    return [load_one_model(g, path, args) for g, path in specs]


def preprocess_bgr(image_bgr: np.ndarray, h: int, w: int, ordering: str) -> np.ndarray:
    x = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    x = x[:, :, ::-1]
    if ordering == "channels_first":
        x = np.rollaxis(x, 2, 0)
    return x


def predict_classmap(item: dict, image_bgr: np.ndarray) -> np.ndarray:
    model = item["model"]
    x = preprocess_bgr(image_bgr, item["input_h"], item["input_w"], item["ordering"])
    y = model(np.expand_dims(x, axis=0), training=False).numpy()

    if isinstance(y, list):
        y = y[0]

    y = y[0]
    n_classes = item["n_classes"]

    if y.ndim == 3:
        pr = np.argmax(y, axis=-1)
    elif y.ndim == 2:
        pr = y.reshape((item["output_h"], item["output_w"], n_classes)).argmax(axis=2)
    else:
        raise RuntimeError(f"Unsupported model output shape for G={item['G']}: {y.shape}")

    return pr.astype(np.uint8)


def colormap_id(name: str) -> int:
    return {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "HOT": cv2.COLORMAP_HOT,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    }[name]


def consensus_from_models(models, image_bgr: np.ndarray, out_size: Tuple[int, int], fire_class: int):
    out_w, out_h = out_size
    acc = np.zeros((out_h, out_w), dtype=np.float32)
    per_model = []

    for item in models:
        pr = predict_classmap(item, image_bgr)
        if pr.shape != (out_h, out_w):
            pr = cv2.resize(pr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        mask = (pr == fire_class).astype(np.float32)
        acc += mask
        per_model.append((item["G"], mask))

    consensus = acc / float(len(models))
    gray = np.clip(consensus * 255.0, 0, 255).astype(np.uint8)
    return consensus, gray, per_model


def final_mask(consensus: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(consensus >= threshold, 255, 0).astype(np.uint8)


def overlay_bgr(image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def run_images(args: argparse.Namespace, models) -> None:
    if not args.image:
        raise ValueError("--image is required for --mode images")

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = image.shape[:2]
    stem = Path(args.image).stem

    consensus, gray, per_model = consensus_from_models(models, image, (w, h), args.fire_class)
    color = cv2.applyColorMap(gray, colormap_id(args.colormap))
    mask = final_mask(consensus, args.vote_threshold)

    cv2.imwrite(str(out_dir / f"{stem}_consensus_gray.png"), gray)
    cv2.imwrite(str(out_dir / f"{stem}_consensus_color.png"), color)
    cv2.imwrite(str(out_dir / f"{stem}_final_mask.png"), mask)

    if args.overlay:
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay_bgr(image, color, args.overlay_alpha))

    if args.save_model_masks:
        mask_dir = out_dir / f"{stem}_per_model_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for g, m in per_model:
            cv2.imwrite(str(mask_dir / f"{stem}_G{g}_mask.png"), (m * 255).astype(np.uint8))

    print(f"Saved outputs to: {out_dir}")


def open_writer(path: Path, fps: float, size: Tuple[int, int], codec: str):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size, True)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def run_videos(args: argparse.Namespace, models) -> None:
    if not args.video:
        raise ValueError("--video is required for --mode videos")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = src_w - (src_w % 2)
    out_h = src_h - (src_h % 2)

    vw_color = open_writer(out_dir / "consensus_color.mp4", fps, (out_w, out_h), args.codec)
    vw_gray = open_writer(out_dir / "consensus_gray.mp4", fps, (out_w, out_h), args.codec)
    vw_mask = open_writer(out_dir / "final_mask.mp4", fps, (out_w, out_h), args.codec)
    vw_overlay = open_writer(out_dir / "consensus_overlay.mp4", fps, (out_w, out_h), args.codec) if args.overlay else None

    frames_dir = out_dir / "frames_color"
    if args.save_video_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    stride = max(1, args.stride)
    processed = 0
    seen = 0

    pbar = tqdm(total=total, desc="Aggregating video")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if seen % stride != 0:
            seen += 1
            if total:
                pbar.update(1)
            continue

        if frame.shape[:2] != (out_h, out_w):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        consensus, gray, _ = consensus_from_models(models, frame, (out_w, out_h), args.fire_class)
        color = cv2.applyColorMap(gray, colormap_id(args.colormap))
        mask_bgr = cv2.cvtColor(final_mask(consensus, args.vote_threshold), cv2.COLOR_GRAY2BGR)

        vw_color.write(color)
        vw_gray.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        vw_mask.write(mask_bgr)

        if vw_overlay is not None:
            vw_overlay.write(overlay_bgr(frame, color, args.overlay_alpha))

        if args.save_video_frames:
            cv2.imwrite(str(frames_dir / f"frame_{processed:06d}.png"), color)

        processed += 1
        seen += 1

        if total:
            pbar.update(1)

        if args.max_frames > 0 and processed >= args.max_frames:
            break

    pbar.close()
    cap.release()
    vw_color.release()
    vw_gray.release()
    vw_mask.release()
    if vw_overlay is not None:
        vw_overlay.release()

    print(f"Processed frames: {processed}")
    print(f"Saved outputs to: {out_dir}")


def main() -> None:
    args = parse_args()
    configure_tf()

    specs = parse_model_specs(args.model)
    models = load_models(specs, args)

    if args.mode == "images":
        run_images(args, models)
    else:
        run_videos(args, models)


if __name__ == "__main__":
    main()
