import numpy as np
import cv2
from tqdm import tqdm

from utils import (
    open_video,
    load_video_frames,
    save_video
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
    cap, width, height, fps = open_video(stabilized_video_path)
    frames = load_video_frames(cap)
    cap, _, _, _ = open_video(binary_mask_path)
    masks = load_video_frames(cap)

    if len(frames) != len(masks):
        raise ValueError("Mismatch between number of frames and masks.")

    # Load and resize background
    background = cv2.imread(background_path)
    if background is None:
        raise FileNotFoundError(f"Background image not found: {background_path}")
    background_resized = cv2.resize(background, (width, height))

    matted_frames = []
    alpha_frames = []

    for frame, mask in tqdm(zip(frames, masks), desc="Video Matting", total=len(frames), unit="frame"):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        alpha = smooth_alpha(mask)
        blended = blend(frame, background_resized, alpha)

        alpha_display = (alpha * 255).astype(np.uint8)
        alpha_display_bgr = cv2.cvtColor(alpha_display, cv2.COLOR_GRAY2BGR)

        matted_frames.append(blended)
        alpha_frames.append(alpha_display_bgr)

    save_video(matted_output_path, matted_frames, fps, (width, height), True)
    save_video(alpha_output_path, alpha_frames, fps, (width, height), True)

    logger.debug("Video matting completed.")
    return matted_frames, alpha_frames
