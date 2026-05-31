"""
Common utility functions for the MapleStory auto-detection system.
Extracted and adapted from MapleStoryAutoLevelUp-main/src/utils/common.py
"""
import os
import datetime
import platform
import time

import cv2
import numpy as np
import yaml

from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.logger import logger

OS_NAME = platform.system()


def is_mac() -> bool:
    return OS_NAME == "Darwin"


def is_windows() -> bool:
    return OS_NAME == "Windows"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    """Load a YAML file and convert lists to tuples for hashability."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return convert_lists_to_tuples(data)


def convert_lists_to_tuples(obj):
    """Recursively convert lists to tuples (for dict keys, etc.)."""
    if isinstance(obj, dict):
        return {convert_lists_to_tuples(k): convert_lists_to_tuples(v)
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return tuple(convert_lists_to_tuples(item) for item in obj)
    return obj


def convert_tuples_to_lists(obj):
    """Recursively convert tuples to lists (for YAML serialization)."""
    if isinstance(obj, dict):
        return {convert_tuples_to_lists(k): convert_tuples_to_lists(v)
                for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [convert_tuples_to_lists(item) for item in obj]
    return obj


def override_cfg(base: dict, override: dict) -> dict:
    """In-place recursive override of base with keys from override."""
    if override is None:
        return base
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            override_cfg(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str, mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Load an image from disk with existence and validity checks.

    Args:
        path: Path to the image file.
        mode: OpenCV read mode (cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, etc.)

    Returns:
        numpy array of the image.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be loaded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, mode)
    if img is None:
        raise ValueError(f"Failed to load image (may be corrupted): {path}")
    return img


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def get_mask(img: np.ndarray, ignore_pixel_color: tuple) -> np.ndarray:
    """
    Create a binary mask where pixels matching ignore_pixel_color are 0 (ignored),
    all others are 255 (included). Used to mask out background color during
    template matching.

    Args:
        img: Input BGR image.
        ignore_pixel_color: BGR color tuple to ignore.

    Returns:
        Single-channel uint8 mask (0 = ignore, 255 = include).
    """
    mask = np.where(np.all(img == ignore_pixel_color, axis=2), 0, 255).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------

def find_pattern_sqdiff(
    img: np.ndarray,
    img_pattern: np.ndarray,
    last_result: tuple = None,
    mask: np.ndarray = None,
    local_search_radius: int = 50,
    global_threshold: float = 0.4
) -> tuple:
    """
    Template matching using cv2.TM_SQDIFF_NORMED with two-phase search.

    Phase 1 (local): If last_result is provided, searches a region around
    the last known position. If the score is below global_threshold, returns
    immediately (cache hit).

    Phase 2 (global): Falls back to full-image search.

    Args:
        img: The search image (grayscale or color).
        img_pattern: The template to find.
        last_result: (x, y) of the last known match location, or None.
        mask: Optional mask for the template.
        local_search_radius: Radius in pixels around last_result to search.
        global_threshold: Score threshold for the local search fast path.

    Returns:
        (min_loc, min_val, is_cached) where:
            - min_loc: (x, y) top-left of the best match
            - min_val: the match score (lower = better)
            - is_cached: True if found via local search, False if via global
    """
    # Check template vs image size
    h_img, w_img = img.shape[:2]
    h_pat, w_pat = img_pattern.shape[:2]
    if h_pat > h_img or w_pat > w_img:
        return (0, 0), 1.0, False

    # Phase 1: Local search (fast path)
    if last_result is not None:
        x0 = max(0, last_result[0] - local_search_radius)
        y0 = max(0, last_result[1] - local_search_radius)
        x1 = min(w_img - w_pat, last_result[0] + local_search_radius)
        y1 = min(h_img - h_pat, last_result[1] + local_search_radius)

        if x1 > x0 and y1 > y0:
            roi = img[y0:y1 + h_pat, x0:x1 + w_pat]
            result = cv2.matchTemplate(
                roi, img_pattern, cv2.TM_SQDIFF_NORMED, mask=mask
            )

            # Handle NaN/Inf in result
            result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)

            min_val, _, min_loc_inner, _ = cv2.minMaxLoc(result)
            min_loc = (min_loc_inner[0] + x0, min_loc_inner[1] + y0)

            if min_val <= global_threshold:
                return min_loc, min_val, True

    # Phase 2: Global fallback
    result = cv2.matchTemplate(img, img_pattern, cv2.TM_SQDIFF_NORMED, mask=mask)
    result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
    min_val, _, min_loc, _ = cv2.minMaxLoc(result)

    return min_loc, min_val, False


# ---------------------------------------------------------------------------
# Non-Maximum Suppression (NMS)
# ---------------------------------------------------------------------------

def nms(monsters: list, iou_threshold: float = 0.3) -> list:
    """
    Apply Non-Maximum Suppression to remove overlapping monster detections.

    Each monster is a dict with keys: "position" (x, y), "size" (h, w), "score".

    Args:
        monsters: List of monster detection dicts.
        iou_threshold: IoU above which boxes are considered overlapping.

    Returns:
        Filtered list of monsters.
    """
    if not monsters:
        return []

    # Sort by score ascending (lower SQDIFF score = better match)
    sorted_monsters = sorted(monsters, key=lambda m: m["score"])
    keep = []

    while sorted_monsters:
        best = sorted_monsters.pop(0)
        keep.append(best)

        # Remove overlapping boxes
        filtered = []
        for m in sorted_monsters:
            if get_iou(
                (best["position"][0], best["position"][1],
                 best["position"][0] + best["size"][1],
                 best["position"][1] + best["size"][0]),
                (m["position"][0], m["position"][1],
                 m["position"][0] + m["size"][1],
                 m["position"][1] + m["size"][0])
            ) < iou_threshold:
                filtered.append(m)
        sorted_monsters = filtered

    return keep


def get_iou(box1: tuple, box2: tuple) -> float:
    """
    Calculate Intersection over Union for two bounding boxes.

    Args:
        box1, box2: (x1, y1, x2, y2) format.

    Returns:
        IoU value [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Debug drawing
# ---------------------------------------------------------------------------

def draw_rectangle(
    img: np.ndarray,
    top_left: tuple,
    size: tuple,
    color: tuple,
    text: str,
    thickness: int = 2,
    text_height: float = 0.7
):
    """
    Draw a rectangle and optional label on an image.

    Args:
        img: The image to draw on (modified in-place).
        top_left: (x, y) of the rectangle.
        size: (height, width) of the rectangle.
        color: BGR color tuple.
        text: Label text (empty string to skip).
        thickness: Rectangle line thickness.
        text_height: Font scale for the label.
    """
    x, y = top_left
    h, w = size
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if text:
        cv2.putText(
            img, text, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, text_height, color, thickness
        )


def screenshot(img: np.ndarray, suffix: str = "screenshot"):
    """
    Save an image to the screenshot/ directory with a timestamp filename.

    Args:
        img: The image to save.
        suffix: Suffix for the filename.
    """
    os.makedirs("screenshot", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join("screenshot", f"{timestamp}_{suffix}.png")
    cv2.imwrite(path, img)
    logger.info(f"[screenshot] Saved to {path}")


# ---------------------------------------------------------------------------
# HSV color conversion
# ---------------------------------------------------------------------------

def to_opencv_hsv(color_hsv: list) -> np.ndarray:
    """
    Convert standard HSV (H: 0-360, S: 0-100, V: 0-100) to OpenCV HSV
    (H: 0-179, S: 0-255, V: 0-255).

    Args:
        color_hsv: [H, S, V] in standard range.

    Returns:
        numpy array [H, S, V] in OpenCV range.
    """
    return np.array([
        int(color_hsv[0] / 2),          # H: 360 -> 180
        int(color_hsv[1] * 2.55),       # S: 100 -> 255
        int(color_hsv[2] * 2.55)        # V: 100 -> 255
    ])


# ---------------------------------------------------------------------------
# Window management (Windows)
# ---------------------------------------------------------------------------

if is_windows():
    import win32gui
    import win32con


    def activate_game_window(window_title: str):
        """
        Bring the game window to the foreground.

        Args:
            window_title: Title of the window to activate.
        """
        try:
            hwnd = win32gui.FindWindow(None, window_title)
            if hwnd:
                win32gui.SetForegroundWindow(hwnd)
            else:
                logger.warning(f"[activate_game_window] Window not found: {window_title}")
        except Exception as e:
            logger.warning(f"[activate_game_window] Failed: {e}")


    def get_game_window_title_by_token(token: str) -> str:
        """
        Enumerate all windows and return the title of the first window
        whose title contains the given token (case-insensitive).

        Args:
            token: Substring to search for in window titles.

        Returns:
            The full window title, or empty string if not found.
        """
        result = ""

        def callback(hwnd, _):
            nonlocal result
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if token.lower() in title.lower():
                    result = title
            return True

        win32gui.EnumWindows(callback, None)
        return result


    def resize_window(window_title: str, width: int = 1296, height: int = 759):
        """
        Resize and reposition a window.

        Args:
            window_title: Title of the window to resize.
            width: Target width in pixels.
            height: Target height in pixels.
        """
        try:
            hwnd = win32gui.FindWindow(None, window_title)
            if hwnd:
                win32gui.MoveWindow(hwnd, 0, 0, width, height, True)
                logger.info(f"[resize_window] Resized '{window_title}' to {width}x{height}")
            else:
                logger.warning(f"[resize_window] Window not found: {window_title}")
        except Exception as e:
            logger.warning(f"[resize_window] Failed: {e}")

else:
    def activate_game_window(window_title: str):
        logger.warning("[activate_game_window] Not implemented on this platform")


    def get_game_window_title_by_token(token: str) -> str:
        logger.warning("[get_game_window_title_by_token] Not implemented on this platform")
        return ""


    def resize_window(window_title: str, width: int = 1296, height: int = 759):
        logger.warning("[resize_window] Not implemented on this platform")


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------

def normalize_pixel_coordinate(coord: tuple, window_size: tuple) -> tuple:
    """
    Scale pixel coordinates from the given window_size to the standard
    working size.

    Args:
        coord: (x, y) or (y, x) coordinate in the source window.
        window_size: (height, width) of the source window.

    Returns:
        Scaled coordinate as (x, y).
    """
    src_h, src_w = window_size
    dst_h, dst_w = WINDOW_WORKING_SIZE  # (1296, 700)

    if src_h == dst_h and src_w == dst_w:
        return coord

    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    return (int(coord[0] * scale_x), int(coord[1] * scale_y))


def is_img_16_to_9(img: np.ndarray, cfg: dict) -> bool:
    """
    Check if the image aspect ratio is approximately 16:9.

    Args:
        img: Input image.
        cfg: Config dict with game_window.ratio_tolerance.

    Returns:
        True if the aspect ratio is within tolerance of 16:9.
    """
    h, w = img.shape[:2]
    ratio = w / h
    expected = 16.0 / 9.0
    tolerance = cfg.get("game_window", {}).get("ratio_tolerance", 0.08)
    return abs(ratio - expected) <= tolerance


# ---------------------------------------------------------------------------
# Minimap detection (adapted from reference project common.py)
# ---------------------------------------------------------------------------

def get_minimap_loc_size(img_frame: np.ndarray):
    """
    Detects the minimap location and size in the game frame.

    Finds white-bordered rectangles via connected components analysis.
    The minimap has 1px white borders on all four sides.

    Returns:
        (x, y, w, h) or None if not found.
    """
    white = np.array([255, 255, 255])
    mask_white = cv2.inRange(img_frame, white, white)
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(mask_white, connectivity=8)

    for i in range(1, num_labels):
        x0, y0, rw, rh, area = stats[i]
        if rw < 100 or rh < 100:
            continue
        x1 = x0 + rw - 1
        y1 = y0 + rh - 1

        # Check 1px white borders on all sides
        if not (np.all(img_frame[y0, x0:x0 + rw] == white)
                and np.all(img_frame[y1, x0:x0 + rw] == white)):
            continue
        if not (np.all(img_frame[y0:y0 + rh, x0] == white)
                and np.all(img_frame[y0:y0 + rh, x1] == white)):
            continue

        # Bounding box of non-white content inside the white border
        mask_nonwhite = np.any(
            img_frame[y0:y0 + rh, x0:x0 + rw] != white, axis=2
        ).astype(np.uint8)
        coords = cv2.findNonZero(mask_nonwhite)
        if coords is None:
            continue
        x_m, y_m, w_m, h_m = cv2.boundingRect(coords)
        return (x_m + x0, y_m + y0, w_m, h_m)

    return None


def get_player_location_on_minimap(
    img_minimap: np.ndarray,
    player_color=(136, 255, 255)
):
    """
    Find the player's position on the minimap by color matching.

    Args:
        img_minimap: Minimap image ROI.
        player_color: BGR color of the player dot on minimap.

    Returns:
        (x, y) center of player dot, or None.
    """
    mask = cv2.inRange(img_minimap, player_color, player_color)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 4:
        return None
    avg = coords.mean(axis=0)[0]
    return (int(round(avg[0])), int(round(avg[1])))


def get_all_other_player_locations_on_minimap(
    img_minimap: np.ndarray,
    red_bgr=(0, 0, 255)
):
    """
    Detect other players as red dots on the minimap.

    Tries increasing color tolerance to find red pixels.

    Returns:
        List of (x, y) tuples.
    """
    red_bgr = tuple(map(int, red_bgr))
    for tolerance in [10, 20, 30, 40]:
        lower = tuple(max(0, c - tolerance) for c in red_bgr)
        upper = tuple(min(255, c + tolerance) for c in red_bgr)
        mask = cv2.inRange(img_minimap, lower, upper)
        coords = cv2.findNonZero(mask)
        if coords is not None and len(coords) >= 3:
            return [tuple(pt[0]) for pt in coords]
    return []
