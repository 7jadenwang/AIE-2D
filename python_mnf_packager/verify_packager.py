"""Exercise real encrypted MNFs with grayscale, quartet, and full-DMD inputs."""
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import package_dmd_mnf as packager


def run_cli(*arguments, success=True):
    result = subprocess.run([sys.executable, str(packager.ROOT / "package_dmd_mnf.py"),
                             *map(str, arguments)], capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError(result.stdout + result.stderr)
    return result


def read_layer(archive, index):
    with Image.open(io.BytesIO(archive.read(f"S{index:06d}_P1.png"))) as image:
        return np.asarray(image).copy()


def main():
    with tempfile.TemporaryDirectory(prefix="verify_dmd_mnf_") as temporary:
        root = Path(temporary)
        inputs = root / "inputs"
        inputs.mkdir()
        quartet = np.zeros((320, 512), dtype=np.uint8)
        for x, y in [(128, 80), (384, 80), (128, 240), (384, 240)]:
            quartet[y - 5:y + 5, x - 5:x + 5] = np.arange(1, 101, dtype=np.uint8).reshape(10, 10)
        full = np.zeros((1600, 2560), dtype=np.uint8)
        full[710:730, 1654:1674] = 73
        gray = np.arange(1, 256, dtype=np.uint8).reshape(15, 17)
        samples = {"quartet": (quartet, (1536, 640)),
                   "full": (full, (0, 0)), "gray": (gray, (1784, 793))}
        for name, (source, origin) in samples.items():
            Image.fromarray(source).save(inputs / f"{name}.png")
        output = root / "output"
        run_cli("--input", inputs, "--output", output)
        for name, (source, (x, y)) in samples.items():
            expected = np.zeros((1600, 2560), dtype=np.uint8)
            height, width = source.shape
            expected[y:y + height, x:x + width] = source
            with zipfile.ZipFile(output / f"{name}.mnf") as archive:
                archive.setpassword(packager.PASSWORD.encode("ascii"))
                assert np.array_equal(read_layer(archive, 1), expected)
                assert np.array_equal(read_layer(archive, 2), expected)
                ys, xs = np.nonzero(expected)
                expected_blast = np.zeros_like(expected)
                expected_blast[max(0, ys.min() - 20):min(1600, ys.max() + 21),
                               max(0, xs.min() - 20):min(2560, xs.max() + 21)] = 43
                assert np.array_equal(read_layer(archive, 3), expected_blast)
                buildscript = archive.read("buildscript.ini").decode()
                assert "number of slices = 2" in buildscript
                assert "number of override slices = 1" in buildscript
                assert "override illumination time = 10.0" in buildscript
                assert "0.1000, S000002_P1, 0.5" in buildscript
                assert "0.2000, S000003_P1, 0.5" in buildscript
        run_cli("--input", inputs, "--output", output, success=False)
        run_cli(inputs / "gray.png", "--output", root / "single", "--center", 1664, 720, "--no-blast")
        with zipfile.ZipFile(root / "single" / "gray.mnf") as archive:
            archive.setpassword(packager.PASSWORD.encode("ascii"))
            assert "S000003_P1.png" not in archive.namelist()
            assert np.array_equal(read_layer(archive, 1)[713:728, 1656:1673], gray)
        run_cli(inputs / "gray.png", "--output", root / "clipped", "--top-left", 2559, 1599, success=False)
        assert not (root / "clipped").exists()
        records = json.loads((output / "manifest.json").read_text())["outputs"]
        assert len(records) == 3 and all(record["verified"] for record in records)
        # Keep one representative preview for visual inspection.
        preview = packager.text_and_shape_preview(
            "prepared_grayscale_snowflake_quartet_2xprojection_blast_gray043_buffer20px.mnf",
            np.pad(quartet, ((640, 640), (1536, 512))),
            packager.load_helper("mono_project_builder"))
        packager.save_png(packager.ROOT / "verification_preview.png", preview)
    print("PASS: grayscale preservation, padding, repeated layers, blast skirt, encryption, exposures,")
    print("      full-DMD coordinates, no-blast mode, collision protection, and cut-off rejection.")


if __name__ == "__main__":
    main()
