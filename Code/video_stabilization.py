import numpy as np
import cv2
from tqdm import tqdm

from utils import (
    open_video,
    load_video_frames,
    close_video,
    save_video,
    smooth_trajectory,
    scale_frame_border
)
from paths import STABILIZED_VIDEO_PATH, TRANSFORMS_PATH
from logger import get_logger

logger = get_logger()

# Optical flow parameters
MAX_FEATURES = 500
FEATURE_QUALITY = 0.01
MIN_FEATURE_DISTANCE = 30
FEATURE_BLOCK_SIZE = 3

# Smoothing parameters
TRAJECTORY_SMOOTH_RADIUS = 5


def stabilize_video(input_video_path):
    """
    Stabilizes the input video using optical flow and homography estimation.
    Outputs a stabilized video and a transform matrix file.

    Args:
        input_video_path (str): Path to the input (unstabilized) video.
    """
    logger.info("Video Stabilization - Starting Process")

    # Open input video and load all frames
    cap, width, height, fps = open_video(input_video_path)
    frames = load_video_frames(cap, color_space='bgr')
    close_video(cap)

    num_frames = len(frames)

    # Prepare tracking on first frame
    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize storage for transform matrices
    raw_transforms_flat = np.zeros((num_frames - 1, 9), dtype=np.float32)
    smoothed_homographies = np.zeros((num_frames - 1, 3, 3), dtype=np.float32)

    # Estimate frame-to-frame transforms using optical flow and homography
    for i, curr_frame in tqdm(enumerate(frames[1:]), total=num_frames - 1, desc="Estimating transforms"):
        prev_features = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=MAX_FEATURES,
            qualityLevel=FEATURE_QUALITY,
            minDistance=MIN_FEATURE_DISTANCE,
            blockSize=FEATURE_BLOCK_SIZE
        )

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_features, None)

        valid_idx = np.where(status == 1)[0]
        prev_pts = prev_features[valid_idx]
        curr_pts = curr_features[valid_idx]

        transform_matrix, _ = cv2.findHomography(prev_pts, curr_pts)
        raw_transforms_flat[i] = transform_matrix.flatten()

        prev_gray = curr_gray

    # Smooth trajectory
    trajectory = np.cumsum(raw_transforms_flat, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory, TRAJECTORY_SMOOTH_RADIUS)
    delta = smoothed_trajectory - trajectory
    corrected_transforms_flat = raw_transforms_flat + delta

    # Apply transforms to stabilize video
    stabilized_frames = [frames[0]]
    for i, frame in tqdm(enumerate(frames[:-1]), total=num_frames - 1, desc="Warping frames"):
        matrix = corrected_transforms_flat[i].reshape((3, 3))
        stabilized = cv2.warpPerspective(frame, matrix, (width, height))
        stabilized = scale_frame_border(stabilized)
        stabilized_frames.append(stabilized)
        smoothed_homographies[i] = matrix

    # Save results
    save_video(STABILIZED_VIDEO_PATH, stabilized_frames, fps, (width, height), is_color=True)
    logger.info(f"Stabilized video saved to {STABILIZED_VIDEO_PATH}")
    smoothed_homographies.dump(TRANSFORMS_PATH)

    logger.info("Video Stabilization - Completed Successfully")