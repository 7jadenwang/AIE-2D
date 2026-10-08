"""Package prepared grayscale images using the established Phoebe MNF format."""
from __future__ import annotations

import argparse
import configparser
import importlib.util
import io
import json
import math
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
CANVAS_SIZE = (2560, 1600)
PASSWORD = "MONO129034"
PREVIEW_SIZE = 200
SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}


def load_helper(name: str):
    path = ROOT / "helpers" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_source(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"{path.name}: use a single-frame image")
        if image.mode == "1":
            image = image.convert("L")
        elif image.mode in {"RGB", "RGBA"}:
            channels = np.asarray(image)
            if not (np.array_equal(channels[..., 0], channels[..., 1])
                    and np.array_equal(channels[..., 0], channels[..., 2])):
                raise ValueError(f"{path.name}: source must be grayscale")
            if image.mode == "RGBA" and np.any(channels[..., 3] != 255):
                raise ValueError(f"{path.name}: flatten transparency onto black first")
            return check_source(channels[..., 0].copy(), path)
        if image.mode != "L":
            raise ValueError(f"{path.name}: expected 8-bit grayscale, got {image.mode}")
        return check_source(np.asarray(image).copy(), path)


def check_source(source: np.ndarray, path: Path) -> np.ndarray:
    height, width = source.shape
    if width > CANVAS_SIZE[0] or height > CANVAS_SIZE[1]:
        raise ValueError(f"{path.name}: {width}x{height} exceeds the DMD canvas")
    if not np.any(source):
        raise ValueError(f"{path.name}: source is completely black")
    return source


def pad_source(source: np.ndarray, args) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = source.shape
    if (width, height) == CANVAS_SIZE:
        return source.copy(), (0, 0)
    if args.top_left is not None:
        x, y = args.top_left
    else:
        # Preserve the old quartet origin for 512x320 sources.
        x, y = args.center[0] - width // 2, args.center[1] - height // 2
    if x < 0 or y < 0 or x + width > CANVAS_SIZE[0] or y + height > CANVAS_SIZE[1]:
        raise ValueError(f"Placement {(x, y)} would cut off the {width}x{height} source")
    projection = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0]), dtype=np.uint8)
    projection[y:y + height, x:x + width] = source
    return projection, (x, y)


def content_bbox(projection: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(projection)
    if points is None:
        raise ValueError("Projection is empty")
    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width, y + height


def make_blast(projection: np.ndarray, buffer: int, gray: int):
    x1, y1, x2, y2 = content_bbox(projection)
    bbox = (max(0, x1 - buffer), max(0, y1 - buffer),
            min(CANVAS_SIZE[0], x2 + buffer), min(CANVAS_SIZE[1], y2 + buffer))
    blast = np.zeros_like(projection)
    blast[bbox[1]:bbox[3], bbox[0]:bbox[2]] = gray
    return blast, bbox


def save_png(path: Path, image: np.ndarray):
    Image.fromarray(image).save(path)


def text_and_shape_preview(filename: str, projection: np.ndarray, builder) -> np.ndarray:
    preview = np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    chosen = None
    # Fit the entire filename into the top part without truncating it.
    for scale in np.linspace(0.5, 0.08, 85):
        lines = builder.wrap_text_to_width(filename, font, float(scale), 1, 184)
        (_, height), baseline = cv2.getTextSize("Ag", font, float(scale), 1)
        step = height + baseline + 2
        if len(lines) * step <= 82:
            chosen = lines, float(scale), height, step
            break
    if chosen is None:
        raise ValueError("Filename is too long to fit the preview")
    lines, scale, height, step = chosen
    for index, line in enumerate(lines):
        width = cv2.getTextSize(line, font, scale, 1)[0][0]
        cv2.putText(preview, line, ((200 - width) // 2, 8 + height + index * step),
                    font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    x1, y1, x2, y2 = content_bbox(projection)
    crop = projection[max(0, y1 - 8):min(1600, y2 + 8),
                      max(0, x1 - 8):min(2560, x2 + 8)]
    factor = min(184 / crop.shape[1], 96 / crop.shape[0])
    width = max(1, int(crop.shape[1] * factor))
    height = max(1, int(crop.shape[0] * factor))
    thumbnail = cv2.resize(crop, (width, height), interpolation=cv2.INTER_NEAREST)
    x, y = (200 - width) // 2, 96 + (96 - height) // 2
    preview[y:y + height, x:x + width] = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2BGR)
    return preview


def projection_preview(projection: np.ndarray) -> np.ndarray:
    # Fit the actual DMD frame into the square preview without stretching it.
    preview = np.zeros((200, 200), dtype=np.uint8)
    preview[37:162] = cv2.resize(projection, (200, 125), interpolation=cv2.INTER_AREA)
    return preview


def validate_archive(path: Path, expected_layers: list[np.ndarray], args):
    with zipfile.ZipFile(path) as archive:
        archive.setpassword(PASSWORD.encode("ascii"))
        for index, expected in enumerate(expected_layers, 1):
            name = f"S{index:06d}_P1.png"
            if not archive.getinfo(name).flag_bits & 1:
                raise RuntimeError(f"{name} is not password protected")
            with Image.open(io.BytesIO(archive.read(name))) as image:
                if image.mode != "L" or not np.array_equal(np.asarray(image), expected):
                    raise RuntimeError(f"Packaged pixels differ in {name}")
        actual_layers = sorted(name for name in archive.namelist() if name.startswith("S") and name.endswith("_P1.png"))
        if len(actual_layers) != len(expected_layers):
            raise RuntimeError("Unexpected layer count")
        params = configparser.ConfigParser()
        params.read_string(archive.read("parameters.ini").decode("utf-8"))
        if (params.getint("CURING", "Number of first layers") != 1
                or params.getfloat("CURING", "First layers curing time") != args.base_exposure
                or params.getfloat("CURING", "Normal curing time") != args.normal_exposure):
            raise RuntimeError("Packaged exposure parameters differ")
        for name in ["Preview_Front.png", "Preview_Left.png", "Preview_Right.png", "Preview_Top.png", "Preview_TFR.png"]:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                if image.size != (200, 200):
                    raise RuntimeError(f"Invalid preview size: {name}")


def package_one(path: Path, source: np.ndarray, args, builder, packager) -> dict:
    projection, origin = pad_source(source, args)
    output = args.output / f"{path.stem}.mnf"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    layers = [projection, projection]
    blast_bbox = None
    if not args.no_blast:
        blast, blast_bbox = make_blast(projection, args.blast_buffer, args.blast_gray)
        layers.append(blast)
    with tempfile.TemporaryDirectory(prefix="dmd_mnf_") as temporary:
        temporary_root = Path(temporary)
        slices = temporary_root / "slices"
        slices.mkdir()
        for index, layer in enumerate(layers, 1):
            save_png(slices / f"{index:03d}.png", layer)
        builder.INPUT_SLICES_DIR = str(slices)
        builder.OUTPUT_PROJECT_DIR = str(temporary_root / "projects")
        builder.PROJECT_NAME = path.stem
        builder.PREVIEW_MODE = "TEXT"
        builder.CREATE_MNF_ARCHIVE = False
        builder.NUM_FIRST_LAYERS = 1
        builder.FIRST_LAYERS_CURE_TIME_S = args.base_exposure
        builder.NORMAL_CURE_TIME_S = args.normal_exposure
        builder.main()
        project = temporary_root / "projects" / path.stem
        preview = text_and_shape_preview(output.name, projection, builder)
        for name in ["Front", "Left", "Right", "Top"]:
            save_png(project / f"Preview_{name}.png", preview)
        save_png(project / "Preview_TFR.png", projection_preview(projection))
        errors, warnings = packager.validate_project_folder(project, fix_buildscript=False)
        for warning in warnings:
            print(f"Warning: {warning}")
        if errors:
            raise RuntimeError("; ".join(errors))
        archive_path = temporary_root / f"{path.stem}.mnf"
        files = packager.ordered_files(project, archive_path)
        seven_zip = packager.find_7zip()
        if seven_zip:
            packager.package_with_7zip(seven_zip, project, archive_path, files, PASSWORD, 16)
        else:
            packager.package_with_pyminizip(project, archive_path, files, PASSWORD)
        validate_archive(archive_path, layers, args)
        # Publish only after reading and verifying the encrypted archive.
        staging = args.output / f".{path.stem}.mnf.tmp"
        try:
            staging.write_bytes(archive_path.read_bytes())
            staging.replace(output)
        finally:
            staging.unlink(missing_ok=True)
        save_png(args.output / "dmd_projections" / f"{path.stem}.png", projection)
        if not args.no_blast:
            save_png(args.output / "blast_projections" / f"{path.stem}.png", layers[2])
        save_png(args.output / "qa_previews" / f"{path.stem}.png", preview)
    return {"source": str(path), "mnf": str(output), "source_size_px": list(source.shape[::-1]),
            "paste_top_left_xy": list(origin), "nonzero_pixels": int(np.count_nonzero(projection)),
            "grayscale_levels": int(np.unique(projection).size), "layer_count": len(layers),
            "blast_bbox_xyxy_exclusive": blast_bbox, "verified": True}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="Optional individual source image paths")
    parser.add_argument("--input", type=Path, default=ROOT / "input_images", help="Folder to scan when no image paths are given")
    parser.add_argument("--output", type=Path, default=ROOT / "mnf_output")
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--center", nargs=2, type=int, default=(1792, 800), metavar=("X", "Y"))
    placement.add_argument("--top-left", nargs=2, type=int, metavar=("X", "Y"))
    parser.add_argument("--blast-buffer", type=int, default=20)
    parser.add_argument("--blast-gray", type=int, default=43, help="Default 255/6 rounded half up")
    parser.add_argument("--base-exposure", type=float, default=10.0)
    parser.add_argument("--normal-exposure", type=float, default=0.5)
    parser.add_argument("--no-blast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.blast_buffer < 0 or not 1 <= args.blast_gray <= 255:
        parser.error("Blast buffer must be nonnegative; gray must be 1..255")
    for exposure in (args.base_exposure, args.normal_exposure):
        if not math.isfinite(exposure) or exposure <= 0 or abs(exposure - round(exposure, 1)) > 1e-9:
            parser.error("Exposures must be positive, finite, and in 0.1-second increments (MNF format)")
    return args


def main():
    args = parse_args()
    paths = args.images or sorted(path for path in args.input.glob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found. Put grayscale PNGs in {args.input}, or supply image paths.")
    if len({path.stem.casefold() for path in paths}) != len(paths):
        raise SystemExit("Source filenames must have unique stems to avoid output collisions")
    # Check all sources and placements before creating any packages.
    sources = [(path.resolve(), read_source(path)) for path in paths]
    for path, source in sources:
        pad_source(source, args)
        if (args.output / f"{path.stem}.mnf").exists() and not args.overwrite:
            raise SystemExit(f"{path.stem}.mnf already exists; use --overwrite to replace it")
    for folder in [args.output, args.output / "dmd_projections", args.output / "blast_projections", args.output / "qa_previews"]:
        folder.mkdir(parents=True, exist_ok=True)
    builder, packager = load_helper("mono_project_builder"), load_helper("package_mnf")
    if packager.find_7zip() is None:
        try:
            import pyminizip
        except ImportError as error:
            raise SystemExit("Install 7-Zip or the optional pyminizip fallback to create encrypted MNFs") from error
    records = []
    for index, (path, source) in enumerate(sources, 1):
        print(f"[{index}/{len(sources)}] {path.name}")
        records.append(package_one(path, source, args, builder, packager))
    manifest = {"canvas_px": list(CANVAS_SIZE), "grayscale_policy": "Preserved exactly; no thresholding or resizing",
                "base_layer_count": 1, "base_exposure_s": args.base_exposure,
                "normal_exposure_s": args.normal_exposure, "blast_gray": None if args.no_blast else args.blast_gray,
                "blast_buffer_px": args.blast_buffer, "outputs": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(records)} MNF(s). Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
