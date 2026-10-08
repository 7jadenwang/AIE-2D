import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


DEFAULT_PASSWORD = "MONO129034"
DEFAULT_PREVIEW_ORDER = [
    "Preview_Front.png",
    "Preview_Left.png",
    "Preview_Right.png",
    "Preview_TFR.png",
    "Preview_Top.png",
]


def find_7zip():
    exe = shutil.which("7z") or shutil.which("7z.exe")
    if exe:
        return Path(exe)

    candidates = [
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # 7-Zip records custom installation folders in the Windows registry.
    try:
        import winreg
    except ImportError:
        return None
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, r"SOFTWARE\7-Zip") as key:
                install_dir = winreg.QueryValueEx(key, "Path")[0]
        except OSError:
            continue
        candidate = Path(install_dir) / "7z.exe"
        if candidate.is_file():
            return candidate
    return None


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def validate_project_folder(folder, fix_buildscript=False):
    errors = []
    warnings = []

    buildscript = folder / "buildscript.ini"
    parameters = folder / "parameters.ini"
    if not buildscript.exists():
        errors.append("Missing buildscript.ini")
    if not parameters.exists():
        errors.append("Missing parameters.ini")

    for preview in DEFAULT_PREVIEW_ORDER:
        if not (folder / preview).exists():
            warnings.append(f"Missing preview image: {preview}")

    slice_files = sorted(folder.glob("S*_P1.png"))
    if not slice_files:
        errors.append("No S*_P1.png slice images found")

    if slice_files:
        names = [p.name for p in slice_files]
        expected = [f"S{i:06d}_P1.png" for i in range(1, len(slice_files) + 1)]
        if names != expected:
            warnings.append(
                "Slice filenames are not contiguous from S000001_P1.png: "
                + ", ".join(names[:8])
                + (" ..." if len(names) > 8 else "")
            )

        image_props = []
        for path in slice_files:
            with Image.open(path) as img:
                image_props.append((img.size, img.mode))
        unique_props = sorted(set(image_props))
        if len(unique_props) > 1:
            warnings.append(f"Slice images do not all share size/mode: {unique_props}")

    if buildscript.exists():
        text = read_text(buildscript)
        declared_match = re.search(r"(?im)^number\s+of\s+slices\s*=\s*(\d+)\s*$", text)
        layer_refs = re.findall(r"(?m)^\s*[-+]?\d+(?:\.\d+)?,\s*(S\d{6}_P1)\s*,", text)
        layer_ref_set = set(layer_refs)

        if declared_match:
            declared = int(declared_match.group(1))
            if declared != len(layer_refs):
                msg = (
                    f"buildscript.ini says number of slices = {declared}, "
                    f"but it has {len(layer_refs)} explicit layer line(s)"
                )
                if fix_buildscript:
                    repaired = re.sub(
                        r"(?im)^(number\s+of\s+slices\s*=\s*)\d+(\s*)$",
                        rf"\g<1>{len(layer_refs)}\2",
                        text,
                        count=1,
                    )
                    buildscript.write_text(repaired, encoding="utf-8", newline="")
                    warnings.append("Repaired " + msg)
                else:
                    errors.append(msg + " (rerun with --fix-buildscript to repair)")
        else:
            errors.append("Could not find 'number of slices = ...' in buildscript.ini")

        slice_stems = {p.stem for p in slice_files}
        missing_refs = sorted(layer_ref_set - slice_stems)
        if missing_refs:
            errors.append("buildscript.ini references missing slice image(s): " + ", ".join(missing_refs))

    return errors, warnings


def ordered_files(folder, output_path):
    preferred = ["buildscript.ini", "parameters.ini", *DEFAULT_PREVIEW_ORDER]
    all_files = [p for p in folder.rglob("*") if p.is_file()]

    def include(path):
        if path.resolve() == output_path.resolve():
            return False
        if path.suffix.lower() == ".mnf":
            return False
        return True

    files = []
    used = set()
    for name in preferred:
        path = folder / name
        if path.exists() and include(path):
            files.append(path)
            used.add(path.resolve())

    for path in sorted(folder.glob("S*_P1.png")):
        if include(path) and path.resolve() not in used:
            files.append(path)
            used.add(path.resolve())

    for path in sorted(all_files, key=lambda p: p.relative_to(folder).as_posix().lower()):
        if include(path) and path.resolve() not in used:
            files.append(path)
            used.add(path.resolve())

    return files


def package_with_7zip(seven_zip, folder, output_path, files, password, threads):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", newline="\n") as listfile:
        list_path = Path(listfile.name)
        for path in files:
            listfile.write(path.relative_to(folder).as_posix() + "\n")

    try:
        if output_path.exists():
            output_path.unlink()

        command = [
            str(seven_zip),
            "a",
            "-tzip",
            "-mx=5",
            "-mm=Deflate",
            "-mfb=32",
            f"-mmt={threads}",
            "-mem=ZipCrypto",
            f"-p{password}",
            "-y",
            str(output_path),
            f"@{list_path}",
        ]
        subprocess.run(command, cwd=folder, check=True)
    finally:
        list_path.unlink(missing_ok=True)


def package_with_pyminizip(folder, output_path, files, password):
    import pyminizip

    if output_path.exists():
        output_path.unlink()

    paths = [str(path) for path in files]
    prefixes = [str(path.parent.relative_to(folder)) if path.parent != folder else "" for path in files]
    pyminizip.compress_multiple(paths, prefixes, str(output_path), password, 5)


def main():
    parser = argparse.ArgumentParser(
        description="Package a MonoWare project folder into a .mnf encrypted ZIP archive."
    )
    parser.add_argument("folder", type=Path, help="Folder containing buildscript.ini, parameters.ini, previews, and slices.")
    parser.add_argument("-o", "--output", type=Path, help="Output .mnf path. Defaults to sibling folder-name.mnf.")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="ZipCrypto password.")
    parser.add_argument("--fix-buildscript", action="store_true", help="Repair buildscript slice count before packaging.")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation checks.")
    parser.add_argument("--force-pyminizip", action="store_true", help="Do not use 7-Zip even if found.")
    parser.add_argument("--threads", type=int, default=16, help="7-Zip CPU thread count.")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder does not exist: {folder}")

    output_path = args.output.resolve() if args.output else (folder.parent / f"{folder.name}.mnf")
    if output_path.suffix.lower() != ".mnf":
        output_path = output_path.with_suffix(".mnf")

    if not args.no_validate:
        errors, warnings = validate_project_folder(folder, fix_buildscript=args.fix_buildscript)
        for warning in warnings:
            print("[warning]", warning)
        if errors:
            for error in errors:
                print("[error]", error)
            raise SystemExit("Packaging stopped because validation failed.")

    files = ordered_files(folder, output_path)
    if not files:
        raise SystemExit("No files to package.")

    seven_zip = None if args.force_pyminizip else find_7zip()
    if seven_zip:
        print(f"Using 7-Zip: {seven_zip}")
        package_with_7zip(seven_zip, folder, output_path, files, args.password, args.threads)
    else:
        print("7-Zip CLI not found; using pyminizip fallback.")
        package_with_pyminizip(folder, output_path, files, args.password)

    print(f"Packaged {len(files)} file(s)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
