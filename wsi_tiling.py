import os
import argparse
import numpy as np
from openslide import open_slide
from PIL import Image, ImageDraw


def tile_wsi_guided_by_mask(
    wsi_path,
    mask_array,
    out_dir,
    tile_size=(512, 512),
    overlap=0.0,
    bg_thr=0.10,
    level=0,
    save_overlay=True,
):
    """
    WSI tiling guided by a binary mask (1 = tissue, 0 = background/artifact),
    and generate a thumbnail overlay with tile bounding boxes.

    Args:
        wsi_path (str): Path to the WSI file (.svs, .mrxs, etc.).
        mask_array (np.ndarray): 2D array (H_mask, W_mask) with values {0,1}.
                                 1 = tissue, 0 = background/artifacts.
        out_dir (str): Folder to save accepted tiles (PNG) and overlay thumbnail.
        tile_size (tuple): (tile_height, tile_width) in pixels at `level`.
        overlap (float): Fraction of tile overlap in [0,1).
                         e.g. 0.0 = no overlap, 0.5 = 50% overlap.
        bg_thr (float): Max allowed fraction of background/artifacts (0s) in a tile mask.
                        e.g. 0.10 → skip tiles with >10% background/artifact.
        level (int): WSI pyramid level to read tiles from (0 = full resolution).
    """
    os.makedirs(out_dir, exist_ok=True)

    slide = open_slide(wsi_path)
    wsi_w, wsi_h = slide.level_dimensions[level]  # size at this level

    # --- 1) Compute scaling between WSI level and mask ---
    mask_h, mask_w = mask_array.shape[:2]

    sf_w = wsi_w / mask_w  # how many WSI pixels per mask pixel (x-direction)
    sf_h = wsi_h / mask_h  # how many WSI pixels per mask pixel (y-direction)

    # Optional sanity check: aspect ratio consistency
    ratio_diff = abs((wsi_w / wsi_h) - (mask_w / mask_h))
    if ratio_diff > 0.05:
        print(f"[Warning] Aspect ratios differ WSI vs mask (diff={ratio_diff:.3f}). Check alignment.")

    tile_h, tile_w = tile_size

    # Step between tiles (with overlap)
    step_h = int(tile_h * (1.0 - overlap))
    step_w = int(tile_w * (1.0 - overlap))
    step_h = max(1, step_h)
    step_w = max(1, step_w)

    base_name = os.path.splitext(os.path.basename(wsi_path))[0]
    tile_idx = 0

    # --- 0) Make a thumbnail in the same size as mask for overlay ---
    thumb = slide.get_thumbnail((mask_w, mask_h)).convert("RGB")
    draw = ImageDraw.Draw(thumb)

    for y in range(0, wsi_h - tile_h + 1, step_h):
        for x in range(0, wsi_w - tile_w + 1, step_w):

            # --- 2) Map WSI tile region to mask coordinates ---
            mask_x0 = int(x / sf_w)
            mask_y0 = int(y / sf_h)
            mask_x1 = int((x + tile_w) / sf_w)
            mask_y1 = int((y + tile_h) / sf_h)

            # Clip to mask bounds
            mask_x0 = max(0, min(mask_x0, mask_w - 1))
            mask_y0 = max(0, min(mask_y0, mask_h - 1))
            mask_x1 = max(mask_x0 + 1, min(mask_x1, mask_w))
            mask_y1 = max(mask_y0 + 1, min(mask_y1, mask_h))

            mask_patch = mask_array[mask_y0:mask_y1, mask_x0:mask_x1]

            if mask_patch.size == 0:
                continue

            # --- 3) Compute background/artifact fraction ---
            # mask: 1 = tissue, 0 = bg/artifacts
            tissue_fraction = mask_patch.mean()          # fraction of 1s
            bg_fraction = 1.0 - tissue_fraction          # fraction of 0s

            if bg_fraction > bg_thr:
                # Too much background/artifacts → skip this tile
                continue

            # --- 4) Read WSI tile and save ---
            tile_rgba = slide.read_region((x, y), level, (tile_w, tile_h))
            tile_rgb = tile_rgba.convert("RGB")

            tile_fname = f"{base_name}_x{x}_y{y}_L{level}.png"
            tile_path = os.path.join(out_dir, tile_fname)
            tile_rgb.save(tile_path)
            tile_idx += 1

            # --- 5) Draw bbox on thumbnail (in mask/thumbnail coordinates) ---
            # Rectangle: (left, top, right, bottom)
            draw.rectangle(
                [mask_x0, mask_y0, mask_x1, mask_y1],
                outline=(0, 255, 0),
                width=2
            )

    # Save overlay thumbnail
    if save_overlay:
        overlay_path = os.path.join(out_dir, f"{base_name}_thumbnail_tiles_overlay.png")
        thumb.save(overlay_path)
        print(f"Overlay thumbnail saved to: {overlay_path}")

    print(f"Tiling done. Saved {tile_idx} tiles to: {out_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="WSI tiling guided by segmentation mask.")

    parser.add_argument("--wsi_path", type=str, required=True, help="Path to WSI (.svs/.mrxs).")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to binary mask image (0/1).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for tiles.")
    parser.add_argument("--tile_size", type=int, nargs=2, default=[512, 512], help="Tile size: h w.")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction (0-1).")
    parser.add_argument("--bg_thr", type=float, default=0.10, help="Max allowed background fraction.")
    parser.add_argument("--level", type=int, default=0, help="WSI pyramid level.")
    parser.add_argument("--save_overlay", action="store_true", help="Save thumbnail overlay with tile boxes.")

    args = parser.parse_args()
    # Load mask
    mask_np = np.array(Image.open(args.mask_path).convert("L"))
    mask_bin = (mask_np > 0).astype(np.uint8)

    # Run tiler
    print("Tiling WSI... started...")
    tile_wsi_guided_by_mask(
        wsi_path=args.wsi_path,
        mask_array=mask_bin,
        out_dir=args.out_dir,
        tile_size=tuple(args.tile_size),
        overlap=args.overlap,
        bg_thr=args.bg_thr,
        level=args.level,
        save_overlay=args.save_overlay,
    )