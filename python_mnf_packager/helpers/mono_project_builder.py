import os
import glob
import re
import cv2
import numpy as np
import shutil
import zipfile
import struct

# --- I/O Paths ---
# The folder containing your input PNG slices (e.g., slice_001.png, slice_002.png, etc.)
INPUT_SLICES_DIR = r"C:\Users\jackr\Desktop\AIE Project\20260709_Lshape_withblast_uniform\feeder_slices\test"

# The folder where the complete project structure will be created.
# This folder will be named after your project.
OUTPUT_PROJECT_DIR = r"C:\Users\jackr\Desktop\AIE Project\20260709_Lshape_withblast_uniform"

# --- Project Name ---
PROJECT_NAME = "test"

# --- Machine & Material Profiles (for .ini files) ---
MACHINE_NAME = "pheobe LRS 10"
MATERIALS_PROFILE = "tempo"
BUILD_STRATEGY = "100 microns"

# --- Preview Image ---
# PREVIEW_MODE can be:
# 'TEXT':         Generates a black image with the project name as white text.
# 'IMAGE_FILE':   Uses the image specified in PREVIEW_IMAGE_PATH.
# 'MIDDLE_SLICE': Uses the middle slice of the print job (default).
PREVIEW_MODE = 'TEXT'

# Optional: Path to a single PNG to use for all preview images.
# Only used if PREVIEW_MODE is 'IMAGE_FILE'.
PREVIEW_IMAGE_PATH = None

# --- Slicing & Curing Parameters ---
LAYER_THICKNESS_MICRONS = 100.0
NORMAL_CURE_TIME_S = 30.0

# These are the "override" or "first" layers with different settings.
NUM_FIRST_LAYERS = 2
FIRST_LAYERS_CURE_TIME_S = 30

# --- Build Plate & Lift Parameters ---
Z_LIFT_SPEED_MM_S = 3.0
SUPPORT_BURN_IN_TIME_S = 0.0

# --- Advanced Machine/Print Settings (for parameters.ini) ---
LED_INTENSITY = 30
HIGH_VISCOSITY = "false"
LIGHT_MASK_ENABLE = "false"
THICK_RESIN_HEIGHT_MM = 1.5
THICK_RESIN_BASE_DELAY_S = 20
THICK_RESIN_BASE_COUNT = 5
EXP_STEP_COUNT = 5
PRE_EXP_HOLD_TIME_S = 0

# --- Archive Creation (.mnf) ---
# Set to True to automatically create the password-protected .mnf archive.
# This requires the 'pyzipper' library. Install it by running: 'pip install pyzipper'
# If it fails, set this to False and zip the folder manually.
CREATE_MNF_ARCHIVE = False
MNF_PASSWORD = "MONO129034"
CLEANUP_AFTER_ARCHIVE = True # Deletes the project folder after zipping

# =============================================================================
#  SCRIPT - NO NEED TO EDIT BELOW THIS LINE
# =============================================================================

def natural_sort_key(s):
    """
    Key for natural sorting. e.g. 'image10.png' comes after 'image2.png'.
    It operates on the basename of the path.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', os.path.basename(s))]

def get_slice_info(input_dir):
    """Gathers information about the input slice files."""
    print(f"Reading slices from: {input_dir}")
    # Use glob to find all png files, then sort them naturally to handle names
    # like 'slice_1.png', 'slice_10.png', 'slice_2.png' correctly.
    slice_patterns = ["*.png", "*.jpg", "*.jpeg"]
    slice_paths = []
    for pattern in slice_patterns:
        slice_paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    slice_paths = sorted(slice_paths, key=natural_sort_key)
    
    if not slice_paths:
        print("Error: No .png or .jpg files found in the input directory.")
        return None, None, None, None

    num_slices = len(slice_paths)
    
    # Read the first image to get dimensions
    try:
        first_img = cv2.imread(slice_paths[0], cv2.IMREAD_UNCHANGED)
        if first_img is None:
            raise IOError("First slice image could not be read.")
        height, width = first_img.shape[:2]
    except Exception as e:
        print(f"Error reading first slice for dimensions: {e}")
        return None, None, None, None
        
    print(f"Found {num_slices} slices with dimensions {width}x{height}.")
    return slice_paths, num_slices, width, height

def generate_parameters_ini(output_dir, width, height):
    """Generates the parameters.ini file."""
    
    # Using a dictionary for easier management of parameters
    params = {
        'SELECTED PROFILES': {
            'Machine': MACHINE_NAME,
            'Materials': MATERIALS_PROFILE,
            'Build Stategy': BUILD_STRATEGY,
            'High Viscosity': HIGH_VISCOSITY,
            'Light Mask Enable': LIGHT_MASK_ENABLE,
            'LED Intensity': LED_INTENSITY,
            'ZAxis Lift Speed': Z_LIFT_SPEED_MM_S,
            'ThickResin Height': THICK_RESIN_HEIGHT_MM,
            'ThickResin Base Delay': THICK_RESIN_BASE_DELAY_S,
            'ThickResin Base Count': THICK_RESIN_BASE_COUNT,
            'Exp Step Count': EXP_STEP_COUNT,
            'Pre-Exp Hold Time': PRE_EXP_HOLD_TIME_S,
        },
        'MACHINE SETTINGS': {
            'Platform size X': 19.456,
            'Platform size Y': 12.16,
            'Platform size Z': 130,
            'Image Format': 32,
            'Image size in X': width,
            'Image size in Y': height,
            'Mirror X': 'false',
            'Mirror Y': 'false',
            'Export Mode': 'Encrypted ZIP',
        },
        'SLICING': {
            'Layer Thickness': LAYER_THICKNESS_MICRONS / 1000.0,
        },
        'CURING': {
            'Normal curing time': NORMAL_CURE_TIME_S,
            'Number of first layers': NUM_FIRST_LAYERS,
            'First layers curing time': FIRST_LAYERS_CURE_TIME_S,
        },
        'IMAGE POST PROCESSING': {
            'Method': 1,
        },
        'SCALING': {
            'Scale in X': 1,
            'Scale in Y': 1,
            'Scale in Z': 1,
        },
        'BASE PLATE': {
            'Type': 'None',
            'Height': 0.1,
        },
        'NON SOLID SUPPORT': {
            'Line Thickness': 1,
        },
        'SCAFFOLDING SUPPORT': {
            'Enable support generation': 'false',
        }
    }
    
    content = ""
    for section, values in params.items():
        content += f"[{section}]\n"
        for key, value in values.items():
            content += f"{key} = {value}\n"
        content += "\n"
        
    path = os.path.join(output_dir, "parameters.ini")
    with open(path, 'w') as f:
        f.write(content)
    print("Generated parameters.ini")

def generate_buildscript_ini(output_dir, num_slices):
    """Generates the buildscript.ini file."""
    buildscript_slice_count = max(num_slices - 1, 0)
    
    header = [
        f"Machine = {MACHINE_NAME}",
        f"Slice thickness = {int(LAYER_THICKNESS_MICRONS)}",
        f"number of slices = {buildscript_slice_count}",
        f"illumination time = {NORMAL_CURE_TIME_S:.1f}",
        f"number of override slices = {NUM_FIRST_LAYERS}",
        f"override illumination time = {FIRST_LAYERS_CURE_TIME_S:.1f}",
        f"support burn in time = {SUPPORT_BURN_IN_TIME_S:.1f}",
    ]
    
    layer_lines = []
    slice_thickness_mm = LAYER_THICKNESS_MICRONS / 1000.0    
    
    # MonoWare buildscript lines list layers 2 through N. The first layer image
    # exists in the archive, but it is handled by the header/override settings.
    # The header count must match the number of explicit layer lines.
    for i in range(2, num_slices + 1):
        # The height value in the file seems to be the Z position *at the start* of the layer.
        height_mm = (i - 1) * slice_thickness_mm 
        filename = f"S{i:06d}_P1"
        # Use override time for the first N layers, then normal time.
        cure_time = FIRST_LAYERS_CURE_TIME_S if i <= NUM_FIRST_LAYERS else NORMAL_CURE_TIME_S
        lift_speed = Z_LIFT_SPEED_MM_S
        
        line = f"{height_mm:.4f}, {filename}, {cure_time:.1f}, {lift_speed:.2f}, 0"
        layer_lines.append(line)
        
    content = "\n".join(header + layer_lines)
    
    path = os.path.join(output_dir, "buildscript.ini")
    with open(path, 'w') as f:
        f.write(content)
    print("Generated buildscript.ini")

# Note: The original script had a bug where it passed 'width' and 'height' to this function,
# but the function didn't accept them. This has been corrected.


def process_slice_images(slice_paths, output_dir, width, height):
    """Copies and renames slice images to the standard format."""
    print("Copying and renaming slice images...")
    for i, old_path in enumerate(slice_paths, 1):
        img = cv2.imread(old_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"    [!] Warning: Could not read image {old_path}. Skipping.")
            continue

        # Ensure the image has the correct dimensions, resizing if necessary
        if img.shape[0] != height or img.shape[1] != width:
            print(f"    [!] Warning: Resizing {os.path.basename(old_path)} from {img.shape[1]}x{img.shape[0]} to {width}x{height}.")
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # Ensure the image is 8-bit grayscale as expected by some printers
        if len(img.shape) > 2:
            # If the image is color, convert it to grayscale.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # If the image is not 8-bit, convert it.
        if img.dtype != np.uint8:
            # For 16-bit images, scale down to preserve grayscale levels.
            # This avoids contrast stretching which can look like binarization.
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            else:
                # For other dtypes, fall back to normalizing (contrast stretching).
                print(f"    [!] Warning: Unsupported dtype {img.dtype} for slice. Normalizing to 8-bit.")
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        new_filename = f"S{i:06d}_P1.png"
        new_path = os.path.join(output_dir, new_filename)
        cv2.imwrite(new_path, img)
    print(f"Processed {len(slice_paths)} slice images.")

def wrap_text_to_width(text, font, font_scale, thickness, max_width):
    """Wraps text to fit a pixel width, splitting long filename tokens when needed."""
    raw_words = text.replace("_", " ").split()
    lines = []
    current_line = ""

    for word in raw_words:
        test_line = f"{current_line} {word}".strip()
        text_width = cv2.getTextSize(test_line, font, font_scale, thickness)[0][0]
        if text_width <= max_width:
            current_line = test_line
            continue

        if current_line:
            lines.append(current_line)
            current_line = ""

        chunk = ""
        for char in word:
            test_chunk = f"{chunk}{char}"
            chunk_width = cv2.getTextSize(test_chunk, font, font_scale, thickness)[0][0]
            if chunk_width <= max_width:
                chunk = test_chunk
            else:
                if chunk:
                    lines.append(chunk)
                chunk = char
        current_line = chunk

    if current_line:
        lines.append(current_line)
    return lines

def create_text_preview(project_name):
    """Creates a 200x200 black preview image with text scaled to fit."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    max_width = 184
    max_height = 184

    chosen = None
    for thickness in (2, 1):
        for font_scale in np.linspace(0.8, 0.22, 24):
            lines = wrap_text_to_width(project_name, font, font_scale, thickness, max_width)
            line_height, baseline = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1], cv2.getTextSize("Ag", font, font_scale, thickness)[1]
            line_spacing = line_height + baseline + 3
            total_text_height = len(lines) * line_spacing - 3
            widest_line = max((cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines), default=0)
            if widest_line <= max_width and total_text_height <= max_height:
                chosen = (lines, float(font_scale), thickness, line_height, line_spacing, total_text_height)
                break
        if chosen:
            break

    if chosen is None:
        font_scale = 0.18
        thickness = 1
        lines = wrap_text_to_width(project_name, font, font_scale, thickness, max_width)[:12]
        line_height, baseline = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1], cv2.getTextSize("Ag", font, font_scale, thickness)[1]
        line_spacing = line_height + baseline + 2
        total_text_height = min(max_height, len(lines) * line_spacing - 2)
        chosen = (lines, font_scale, thickness, line_height, line_spacing, total_text_height)

    lines, font_scale, thickness, line_height, line_spacing, total_text_height = chosen
    y_start = max(0, (200 - total_text_height) // 2) + line_height

    for i, line in enumerate(lines):
        text_width = cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        x = max(0, (200 - text_width) // 2)
        y = int(y_start + i * line_spacing)
        if y > 196:
            break
        cv2.putText(img, line, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return img

def generate_preview_images(slice_paths, output_dir):
    """Creates preview images from the slice data."""
    preview_names = [
        "Preview_Front.png", "Preview_Left.png", "Preview_Right.png",
        "Preview_TFR.png", "Preview_Top.png"
    ]
    
    # Start with a blank image as the ultimate fallback
    preview_img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    source_used = "a blank image"
    
    # --- 1. Generate preview based on PREVIEW_MODE ---
    if PREVIEW_MODE.upper() == 'TEXT':
        preview_img_bgr = create_text_preview(PROJECT_NAME)
        source_used = "generated text"

    elif PREVIEW_MODE.upper() == 'IMAGE_FILE':
        if PREVIEW_IMAGE_PATH and os.path.exists(PREVIEW_IMAGE_PATH):
            try:
                preview_img_orig = cv2.imread(PREVIEW_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
                if preview_img_orig is None: raise IOError("Could not read specified preview image.")
                if len(preview_img_orig.shape) > 2 and preview_img_orig.shape[2] == 4:
                    preview_img_orig = cv2.cvtColor(preview_img_orig, cv2.COLOR_BGRA2BGR)
                preview_img_resized = cv2.resize(preview_img_orig, (200, 200), interpolation=cv2.INTER_AREA)
                if len(preview_img_resized.shape) == 2:
                    preview_img_bgr = cv2.cvtColor(preview_img_resized, cv2.COLOR_GRAY2BGR)
                else:
                    preview_img_bgr = preview_img_resized
                source_used = f"specified image: {os.path.basename(PREVIEW_IMAGE_PATH)}"
            except Exception as e:
                print(f"Warning: Could not use specified preview image. Error: {e}. Falling back to blank image.")
        else:
            print(f"Warning: PREVIEW_MODE is 'IMAGE_FILE' but PREVIEW_IMAGE_PATH is invalid. Falling back to blank image.")

    elif PREVIEW_MODE.upper() == 'MIDDLE_SLICE':
        if not slice_paths:
            print("Warning: PREVIEW_MODE is 'MIDDLE_SLICE' but no slices were found. Using blank image.")
        middle_slice_path = slice_paths[len(slice_paths) // 2]
        try:
            preview_img_orig = cv2.imread(middle_slice_path, cv2.IMREAD_GRAYSCALE)
            if preview_img_orig is None: raise IOError(f"Could not read middle slice: {middle_slice_path}")
            preview_img_resized = cv2.resize(preview_img_orig, (200, 200), interpolation=cv2.INTER_AREA)
            preview_img_bgr = cv2.cvtColor(preview_img_resized, cv2.COLOR_GRAY2BGR)
            source_used = f"middle slice: {os.path.basename(middle_slice_path)}"
        except Exception as e:
            print(f"Warning: Could not generate preview from middle slice. Using a blank image. Error: {e}")
    
    # --- 2. Save all previews using the single generated image ---
    print(f"Generating previews from {source_used}.")
    for name in preview_names:
        path = os.path.join(output_dir, name)
        cv2.imwrite(path, preview_img_bgr)

def main():
    """Main function to build the project structure."""
    final_project_path = os.path.join(OUTPUT_PROJECT_DIR, PROJECT_NAME)
    os.makedirs(final_project_path, exist_ok=True)
    print(f"Project will be created in: {final_project_path}")
    
    slice_paths, num_slices, width, height = get_slice_info(INPUT_SLICES_DIR)
    if not slice_paths:
        return
        
    generate_parameters_ini(final_project_path, width, height)
    generate_buildscript_ini(final_project_path, num_slices)
    process_slice_images(slice_paths, final_project_path, width, height) # Pass width and height
    generate_preview_images(slice_paths, final_project_path)
    
    print(f"\nSuccessfully created project folder '{PROJECT_NAME}' with {num_slices} layers.")
    print("You can now manually zip the contents of the folder and rename it to .mnf")

if __name__ == "__main__":
    main()
