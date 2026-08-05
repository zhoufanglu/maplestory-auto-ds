import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.common import load_yaml, override_cfg, get_mask, find_pattern_sqdiff


def build_title_inputs(frame_bgr: np.ndarray, title_bgr: np.ndarray, title_cfg: dict):
    mode = str(title_cfg.get("mode", "grayscale")).lower()
    title_gray = cv2.cvtColor(title_bgr, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    try:
        white_lower = int(title_cfg.get("white_lower", 150))
        white_upper = int(title_cfg.get("white_upper", 255))
        blur_kernel = int(title_cfg.get("blur_kernel", 3))
    except (TypeError, ValueError):
        white_lower, white_upper, blur_kernel = 150, 255, 3

    white_lower = max(0, min(255, white_lower))
    white_upper = max(white_lower, min(255, white_upper))
    if blur_kernel < 3 or blur_kernel % 2 == 0:
        blur_kernel = 0

    if mode == "white_mask":
        if blur_kernel:
            title_gray = cv2.GaussianBlur(title_gray, (blur_kernel, blur_kernel), 0)
            frame_gray = cv2.GaussianBlur(frame_gray, (blur_kernel, blur_kernel), 0)
        title_proc = cv2.inRange(title_gray, white_lower, white_upper)
        frame_proc = cv2.inRange(frame_gray, white_lower, white_upper)
        return frame_proc, title_proc, title_proc, mode

    mask_title = get_mask(title_bgr, (0, 255, 0))
    return frame_gray, title_gray, mask_title, "grayscale"


def main():
    parser = argparse.ArgumentParser(description="Test title-based player localization on a screenshot")
    parser.add_argument("--image", default=os.path.join("test", "test_image", "screenshot.png"))
    parser.add_argument("--cfg", default="custom")
    parser.add_argument("--save", default=os.path.join("screenshot", "title_locator_debug.png"))
    args = parser.parse_args()

    cfg = load_yaml(os.path.join(ROOT, "config", "config_default.yaml"))
    custom_cfg = os.path.join(ROOT, "config", f"config_{args.cfg}.yaml")
    if os.path.exists(custom_cfg):
        cfg = override_cfg(cfg, load_yaml(custom_cfg))

    frame = cv2.imread(os.path.join(ROOT, args.image))
    title = cv2.imread(os.path.join(ROOT, "character", "title.png"))
    if frame is None:
        raise FileNotFoundError(f"Test image not found: {args.image}")
    if title is None:
        raise FileNotFoundError("character/title.png not found")

    title_cfg = cfg.get("title", {})
    search_img, pattern_img, mask_title, mode_used = build_title_inputs(frame, title, title_cfg)
    threshold = float(title_cfg.get("match_threshold", 0.3))
    offset = title_cfg.get("offset", [0, -60])
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        offset = [0, -60]

    loc, score, _ = find_pattern_sqdiff(search_img, pattern_img, mask=mask_title, global_threshold=threshold)
    tw, th = title.shape[1], title.shape[0]
    player = (loc[0] + tw // 2 + int(offset[0]), loc[1] + th + int(offset[1]))
    hit = score < threshold

    debug = frame.copy()
    color = (0, 255, 0) if hit else (0, 0, 255)
    cv2.rectangle(debug, loc, (loc[0] + tw, loc[1] + th), color, 2)
    cv2.circle(debug, player, 6, (255, 0, 0), -1)
    cv2.putText(debug, f"TITLE {mode_used} score={score:.4f}", (max(0, loc[0] - 10), max(20, loc[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(debug, f"PLAYER {player}", (player[0] + 8, max(20, player[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.join(ROOT, args.save)), exist_ok=True)
    cv2.imwrite(os.path.join(ROOT, args.save), debug)

    print(f"mode={mode_used}")
    print(f"threshold={threshold}")
    print(f"score={score:.6f}")
    print(f"hit={hit}")
    print(f"title_loc={loc}")
    print(f"player_loc={player}")
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()

