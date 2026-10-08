# DMD MNF packager

This folder includes the same MonoWare project builder and encrypted ZIP packager used for the earlier packages. Install Python dependencies with `python -m pip install -r requirements.txt`. Install 7-Zip for encrypted MNF archives; the existing optional `pyminizip` fallback also works when available.

Place prepared grayscale PNGs in `input_images`, then run:

```powershell
python 'D:\Research\\AIE 2D\\mnf_packager_oct_05_fullrun\\20261005_python_mnf_packager\\package_dmd_mnf.py'
```

Outputs go in `mnf_output`: one MNF per source, padded DMD PNGs, blast PNGs, filename-and-shape QA previews, and a verification manifest. The source filename becomes the MNF filename.

Default settings:

- Black DMD canvas: 2560 x 1600 px; source pixels preserved without thresholding, resizing, or contrast changes.
- Smaller sources centered at (1792, 800). A 512 x 320 quartet is pasted at (1536, 640), preserving the usual column 7/8, row 5/6 positions.
- Already-full-DMD sources retain their original coordinates.
- First two layers: identical projection.
- Third layer: rectangular nonzero bounding box plus a 20 px skirt, gray 43 (255/6 rounded half up). Skirt clipped to the DMD edge.
- One base layer at 10 seconds; normal layers at 0.5 seconds.
- Front/Left/Right/Top: fitted MNF filename with cropped feature thumbnail. TFR: actual DMD projection fitted to the preview box.
- Same machine/material parameters, password, slice order, and archive format as the existing builder.
- Archives are decrypted and compared to the intended pixels before publishing. Existing MNFs require `--overwrite` to replace them.

Single feature at the usual first cell center:

```powershell
python package_dmd_mnf.py "C:\path\feature.png" --center 1664 720
```

Another source folder:

```powershell
python package_dmd_mnf.py --input "C:\path\sources" --output "C:\path\packages"
```

Custom settings:

```powershell
python package_dmd_mnf.py --top-left 1536 640 --blast-buffer 20 --blast-gray 43 --base-exposure 10 --normal-exposure 0.5
python package_dmd_mnf.py --no-blast
```

Exposure times use 0.1-second increments because the existing buildscript writer uses one decimal place. Inputs must be single-frame 8-bit grayscale images (or RGB images with identical channels). Incompatible inputs and placements that would cut off the source are rejected. Use `python package_dmd_mnf.py --help` for all options.
