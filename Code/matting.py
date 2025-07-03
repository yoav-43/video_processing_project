import numpy as np
import cv2

from utils import (
    get_video_files,
    load_entire_video,
    write_video
)
from logger import get_logger
logger = get_logger()


def smooth_alpha(binary_mask, radius=15):
    """Apply Gaussian blur to binary mask to create soft alpha matte."""
    if binary_mask.max() > 1:
        binary_mask = binary_mask / 255.0
    blurred = cv2.GaussianBlur(binary_mask.astype(np.float32), (radius | 1, radius | 1), 0)
    return np.clip(blurred, 0, 1)


def blend(foreground, background, alpha):
    """Blend foreground and background using the alpha matte."""
    alpha = alpha[..., None]
    return np.clip(alpha * foreground + (1 - alpha) * background, 0, 255).astype(np.uint8)


def video_matting(stabilized_video_path, binary_mask_path, background_path,
                  matted_output_path, alpha_output_path):
    """
    Perform matting using paths to stabilized and binary videos.
    Saves matted video and alpha mask video.
    """
    logger.debug("Starting video matting.")

    # Load videos
    cap, width, height, fps = get_video_files(stabilized_video_path)
    frames = load_entire_video(cap)
    cap, _, _, _ = get_video_files(binary_mask_path)
    masks = load_entire_video(cap)

    if len(frames) != len(masks):
        raise ValueError("Mismatch between number of frames and masks.")

    # Load and resize background
    background = cv2.imread(background_path)
    if background is None:
        raise FileNotFoundError(f"Background image not found: {background_path}")
    background_resized = cv2.resize(background, (width, height))

    matted_frames = []
    alpha_frames = []

    for i, (frame, mask) in enumerate(zip(frames, masks)):
        logger.debug(f"Processing frame {i + 1}/{len(frames)}")

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        alpha = smooth_alpha(mask)
        blended = blend(frame, background_resized, alpha)

        alpha_display = (alpha * 255).astype(np.uint8)
        alpha_display_bgr = cv2.cvtColor(alpha_display, cv2.COLOR_GRAY2BGR)

        matted_frames.append(blended)
        alpha_frames.append(alpha_display_bgr)

    write_video(matted_output_path, matted_frames, fps, (width, height), True)
    write_video(alpha_output_path, alpha_frames, fps, (width, height), False)

    logger.debug("Video matting completed.")
    return matted_frames, alpha_frames