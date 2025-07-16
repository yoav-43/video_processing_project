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


def smooth_alpha_mask(binary_mask, radius=15):
    """
    Apply Gaussian blur to a binary mask to generate a smooth alpha matte.

    Args:
        binary_mask (np.ndarray): Binary mask (grayscale or single-channel).
        radius (int): Kernel size for Gaussian blur.

    Returns:
        np.ndarray: Smoothed alpha matte in range [0, 1].
    """
    if binary_mask.max() > 1:
        binary_mask = binary_mask / 255.0
    blurred_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (radius | 1, radius | 1), 0)
    return np.clip(blurred_mask, 0, 1)


def blend_foreground_background(foreground_img, background_img, alpha_matte):
    """
    Blend foreground and background images using an alpha matte.

    Args:
        foreground_img (np.ndarray): Foreground image (H x W x 3).
        background_img (np.ndarray): Background image (H x W x 3).
        alpha_matte (np.ndarray): Alpha matte in range [0, 1] (H x W).

    Returns:
        np.ndarray: Blended RGB image.
    """
    alpha_expanded = alpha_matte[..., None]
    blended_img = alpha_expanded * foreground_img + (1 - alpha_expanded) * background_img
    return np.clip(blended_img, 0, 255).astype(np.uint8)


def video_matting(
    stabilized_video_path,
    binary_mask_path,
    background_image_path,
    output_matted_video_path,
    output_alpha_video_path
):
    """
    Perform video matting by compositing a foreground video over a static background
    using a binary mask for segmentation. Outputs matted video and alpha mask video.

    Args:
        stabilized_video_path (str): Path to the stabilized input video.
        binary_mask_path (str): Path to the binary mask video.
        background_image_path (str): Path to the background image.
        output_matted_video_path (str): Path to save the final composited video.
        output_alpha_video_path (str): Path to save the alpha mask video.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: List of matted frames and alpha mask frames.
    """
    logger.debug("Video Matting - Starting Process")

    # Load foreground frames
    cap_foreground, width, height, fps = open_video(stabilized_video_path)
    foreground_frames = load_video_frames(cap_foreground)

    # Load binary mask frames
    cap_mask, _, _, _ = open_video(binary_mask_path)
    binary_mask_frames = load_video_frames(cap_mask)

    if len(foreground_frames) != len(binary_mask_frames):
        raise ValueError("Mismatch between number of foreground frames and mask frames.")

    # Load and resize static background
    background_image = cv2.imread(background_image_path)
    if background_image is None:
        raise FileNotFoundError(f"Background image not found: {background_image_path}")
    background_resized = cv2.resize(background_image, (width, height))

    matted_video_frames = []
    alpha_video_frames = []

    for frame, mask in tqdm(zip(foreground_frames, binary_mask_frames),
                             desc="Video Matting", total=len(foreground_frames), unit="frame"):
        # Convert to grayscale if mask is in color
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Create soft alpha mask and blend
        alpha_matte = smooth_alpha_mask(mask)
        matted_frame = blend_foreground_background(frame, background_resized, alpha_matte)

        # Prepare alpha display frame (grayscale -> BGR)
        alpha_display = (alpha_matte * 255).astype(np.uint8)
        alpha_display_bgr = cv2.cvtColor(alpha_display, cv2.COLOR_GRAY2BGR)

        # Store results
        matted_video_frames.append(matted_frame)
        alpha_video_frames.append(alpha_display_bgr)

    # Save output videos
    save_video(output_matted_video_path, matted_video_frames, fps, (width, height), True)
    logger.info(f"Matted video saved to {output_matted_video_path}")

    save_video(output_alpha_video_path, alpha_video_frames, fps, (width, height), True)
    logger.info(f"Alpha mask video saved to {output_alpha_video_path}")

    logger.debug("Video Matting - Completed Successfully")

    return matted_video_frames, alpha_video_frames
