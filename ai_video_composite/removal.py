#  ai-video-composite - Background Removal Module
#
#  AI background removal (rembg isnet-anime), alpha channel cleanup,
#  and green spill defringing via hue rotation.
#
#  Depends on: numpy, Pillow, rembg, scipy (optional)
#  Used by:    compositing.py, __main__.py, __init__.py

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# rembg model — isnet-anime produces much cleaner cutouts for anime/cartoon art
# ---------------------------------------------------------------------------

_rembg_session = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("isnet-anime")
    return _rembg_session


# ---------------------------------------------------------------------------
# Background Removal + Green Spill Cleanup
# ---------------------------------------------------------------------------

def remove_background(img: Image.Image) -> Image.Image:
    """Remove background from an image using rembg (isnet-anime model),
    then clean alpha and green spill.
    Returns RGBA image with stray pixels removed and green fringing cleaned."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    result = remove(img, session=_get_rembg_session())
    result = clean_alpha(result)
    result = defringe_green(result)
    return result


def clean_alpha(
    img: Image.Image,
    alpha_floor: int = 10,
    min_island_pixels: int = 50,
    erode_px: int = 0,
) -> Image.Image:
    """Remove stray semi-transparent pixels and tiny islands from a transparent image.

    Three-step cleanup applied after rembg background removal:
      1. Alpha threshold — pixels below alpha_floor become fully transparent.
      2. Small island removal — connected clusters smaller than min_island_pixels
         are deleted (catches isolated dots/wisps that aren't part of the body).
      3. Light erosion — shrink the alpha mask by erode_px to tighten the silhouette.

    Args:
        img: RGBA PIL Image (output of rembg).
        alpha_floor: Alpha values below this are snapped to 0 (0-255).
        min_island_pixels: Connected components smaller than this are removed.
        erode_px: Pixels to erode from the alpha edge (0 to skip).

    Returns:
        Cleaned RGBA image.
    """
    arr = np.array(img)
    alpha = arr[:, :, 3]

    # --- Step 1: Threshold low-alpha pixels to fully transparent ---
    before_thresh = np.count_nonzero(alpha > 0)
    alpha[alpha < alpha_floor] = 0
    after_thresh = np.count_nonzero(alpha > 0)

    # --- Step 2: Remove small isolated pixel clusters ---
    removed_islands = 0
    try:
        from scipy.ndimage import label
        binary = alpha > 0
        labeled, num_features = label(binary)
        for i in range(1, num_features + 1):
            component = labeled == i
            if np.count_nonzero(component) < min_island_pixels:
                alpha[component] = 0
                removed_islands += 1
    except ImportError:
        pass  # scipy not available, skip island removal

    # --- Step 3: Morphological opening to remove thin wisps/strands ---
    # Opening = erode then dilate. Removes features thinner than the kernel
    # (stray hair outlines, wispy strands) while restoring the bulk shape.
    opened_px = 0
    if erode_px > 0:
        from PIL import ImageFilter
        binary_before = alpha > 0
        alpha_img = Image.fromarray(alpha, mode="L")
        kernel_size = erode_px * 2 + 1
        # Erode (MinFilter) — shrinks everything, thin features disappear
        opened = alpha_img.filter(ImageFilter.MinFilter(size=kernel_size))
        # Dilate (MaxFilter) — grows back, bulk shape restored but thin bits stay gone
        opened = opened.filter(ImageFilter.MaxFilter(size=kernel_size))
        opened_alpha = np.array(opened)
        # Only zero out pixels that the opening removed (don't increase alpha)
        opened_mask = (binary_before) & (opened_alpha == 0)
        opened_px = np.count_nonzero(opened_mask)
        alpha[opened_mask] = 0

    arr[:, :, 3] = alpha
    thresh_removed = before_thresh - after_thresh
    log.info("Clean alpha: %d low-alpha pixels removed, "
             "%d islands removed, %d thin features removed",
             thresh_removed, removed_islands, opened_px)

    return Image.fromarray(arr, "RGBA")


def defringe_green(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """Remove green spill/fringing from edges of a transparent image.

    Instead of just reducing the green channel (which leaves brown), this
    rotates the hue of chroma-green pixels toward the average non-green
    color of the image (typically blonde/skin tone), preserving brightness.

    Args:
        img: RGBA PIL Image with transparency
        strength: 0.0 (no correction) to 1.0 (full correction)

    Returns:
        Cleaned RGBA image.
    """
    arr = np.array(img, dtype=np.float32)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    visible = a > 10

    # --- Compute hue for every pixel ---
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    chroma = max_c - min_c

    hue = np.zeros_like(r)

    mask_chroma = chroma > 1e-6

    g_is_max = mask_chroma & (g >= r) & (g >= b)
    hue[g_is_max] = 60.0 * (((b[g_is_max] - r[g_is_max]) / chroma[g_is_max]) + 2.0)
    r_is_max = mask_chroma & (r > g) & (r >= b)
    hue[r_is_max] = 60.0 * (((g[r_is_max] - b[r_is_max]) / chroma[r_is_max]) % 6.0)
    b_is_max = mask_chroma & (b > r) & (b > g)
    hue[b_is_max] = 60.0 * (((r[b_is_max] - g[b_is_max]) / chroma[b_is_max]) + 4.0)
    hue = hue % 360.0

    # --- Find the target hue from non-green visible pixels ---
    # These are the "natural" colors of the character (skin, hair, clothes)
    non_green = visible & (a > 200) & ~((hue >= 80) & (hue <= 170))
    if np.any(non_green):
        # Weighted average hue of non-green pixels (circular mean)
        hue_rad = np.deg2rad(hue[non_green])
        target_hue = np.degrees(np.arctan2(
            np.mean(np.sin(hue_rad)),
            np.mean(np.cos(hue_rad)),
        )) % 360.0
    else:
        target_hue = 45.0  # Fallback: warm blonde

    # --- Detect chroma green pixels ---
    # Chroma green: hue 80-170° (the actual green screen color and its bleed)
    is_chroma_hue = (hue >= 80) & (hue <= 170)
    green_excess = g - (r + b) / 2.0

    # Edge pixels (semi-transparent) — more aggressive
    is_edge = visible & (a < 240) & (a > 10)
    edge_fix = is_edge & (green_excess > 5) & is_chroma_hue

    # Interior pixels — only clearly green ones
    is_interior = visible & (a >= 240)
    interior_fix = is_interior & (green_excess > 15) & is_chroma_hue

    needs_fix = edge_fix | interior_fix

    if not np.any(needs_fix):
        return img

    # --- Vectorized hue rotation: shift green pixels toward target hue ---
    target_h = target_hue / 360.0

    # Extract only the pixels that need fixing
    fr = r[needs_fix] / 255.0
    fg = g[needs_fix] / 255.0
    fb = b[needs_fix] / 255.0

    # Vectorized RGB → HSV
    fmax = np.maximum(np.maximum(fr, fg), fb)
    fmin = np.minimum(np.minimum(fr, fg), fb)
    fchroma = fmax - fmin

    fh = np.zeros_like(fr)
    fs = np.zeros_like(fr)
    fv = fmax

    chroma_mask = fchroma > 1e-6
    fs[chroma_mask] = fchroma[chroma_mask] / fmax[chroma_mask]

    # Hue calculation per max channel
    r_max = chroma_mask & (fr >= fg) & (fr >= fb)
    g_max = chroma_mask & (fg > fr) & (fg >= fb)
    b_max = chroma_mask & (fb > fr) & (fb > fg)
    fh[r_max] = ((fg[r_max] - fb[r_max]) / fchroma[r_max]) % 6.0 / 6.0
    fh[g_max] = ((fb[g_max] - fr[g_max]) / fchroma[g_max] + 2.0) / 6.0
    fh[b_max] = ((fr[b_max] - fg[b_max]) / fchroma[b_max] + 4.0) / 6.0

    # Blend strength: edges get full shift, interior partial
    blend_arr = np.where(a[needs_fix] < 240, strength, strength * 0.7)

    new_h = (fh + (target_h - fh) * blend_arr) % 1.0
    new_s = np.clip(fs * (1.0 - blend_arr * 0.2), 0.0, 1.0)

    # Vectorized HSV → RGB
    hi = (new_h * 6.0).astype(np.int32) % 6
    f = (new_h * 6.0) - hi.astype(np.float32)
    p = fv * (1.0 - new_s)
    q = fv * (1.0 - f * new_s)
    t = fv * (1.0 - (1.0 - f) * new_s)

    nr = np.zeros_like(fv)
    ng = np.zeros_like(fv)
    nb = np.zeros_like(fv)
    for case, rv, gv, bv in [
        (0, fv, t, p), (1, q, fv, p), (2, p, fv, t),
        (3, p, q, fv), (4, t, p, fv), (5, fv, p, q),
    ]:
        m = hi == case
        nr[m] = rv[m]
        ng[m] = gv[m]
        nb[m] = bv[m]

    arr[needs_fix, 0] = np.clip(nr * 255.0, 0, 255)
    arr[needs_fix, 1] = np.clip(ng * 255.0, 0, 255)
    arr[needs_fix, 2] = np.clip(nb * 255.0, 0, 255)

    edge_count = np.count_nonzero(edge_fix)
    interior_count = np.count_nonzero(interior_fix)
    total = edge_count + interior_count
    log.info("Defringe: %d pixels (%d edge, %d interior) -> target hue %.0f deg",
             total, edge_count, interior_count, target_hue)

    return Image.fromarray(arr.astype(np.uint8), "RGBA")


# ---------------------------------------------------------------------------
# File-level convenience functions
# ---------------------------------------------------------------------------

def standardize_greenscreen(
    input_path: Path,
    output_path: Path,
    green: tuple[int, int, int] = (0, 177, 64),
) -> Path:
    """Convert any-background character image to exact standard chroma green.

    Uses rembg AI to cleanly remove the original background (works regardless
    of background color), then composites the character onto a solid #00b140
    green. This ensures FFmpeg chromakey works reliably — FLUX's pastel sage
    green (~RGB 150,208,166) is too close to character colors for chromakey,
    but standard chroma green is saturated enough to separate cleanly.

    Args:
        input_path: Source character image (any background).
        output_path: Where to save the standardized green-screen image.
        green: RGB tuple for the background (default: #00b140 chroma green).

    Returns:
        Path to saved image.

    Raises:
        FileNotFoundError: If input_path does not exist.
        OSError: If the image cannot be opened or saved.
    """
    try:
        img = Image.open(input_path)
    except (OSError, ValueError) as e:
        raise OSError(f"Cannot open image '{input_path}': {e}") from e

    transparent = remove_background(img)

    # Composite character onto solid chroma green
    bg = Image.new("RGBA", transparent.size, (*green, 255))
    bg.paste(transparent, (0, 0), transparent)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bg.convert("RGB").save(output_path, "PNG")
    log.info("Standardized green screen: %s (%dx%d)", output_path.name, bg.size[0], bg.size[1])
    return output_path


def remove_background_file(input_path: Path, output_path: Path) -> Path:
    """Remove background from a file and save as transparent PNG.

    Raises:
        FileNotFoundError: If input_path does not exist.
        OSError: If the image cannot be opened or saved.
    """
    try:
        img = Image.open(input_path)
    except (OSError, ValueError) as e:
        raise OSError(f"Cannot open image '{input_path}': {e}") from e

    result = remove_background(img)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, "PNG")
    log.info("Background removed: %s (%dx%d)", output_path.name, result.size[0], result.size[1])
    return output_path
