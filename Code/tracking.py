import numpy as np
import cv2
import json
from tqdm import tqdm

from utils import open_video, load_video_frames, save_video
from paths import FINAL_OUTPUT_PATH, TRACKING_JSON_PATH
from logger import get_logger

logger = get_logger()


def track_video(input_video_path: str) -> None:
    """
    Track a moving object in a video using background subtraction and contour detection.

    This function loads a video, applies background subtraction (MOG2),
    detects the largest moving object in each frame, and draws a bounding box around it.
    The output video and tracking results are saved to disk.

    Args:
        input_video_path (str): Path to the input video file.
    """
    logger.info("Video Tracking - Starting Process")

    # Load video and extract properties
    video_capture, frame_width, frame_height, fps = open_video(input_video_path)
    video_frames = load_video_frames(video_capture, color_space='bgr')
    video_capture.release()

    # Create background subtractor (MOG2)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=30, detectShadows=False
    )

    processed_frames = []
    bounding_box_log = {}

    # Initialize default bounding box (centered, arbitrary size)
    box_x, box_y, box_width, box_height = frame_width // 2, frame_height // 2, 100, 200
    last_estimate = np.array([box_x, box_y, box_width, box_height, 0, 0])  # includes velocity

    for frame_index, frame in enumerate(tqdm(video_frames, desc="Tracking: Processing frames")):
        # Generate foreground mask
        foreground_mask = background_subtractor.apply(frame)

        # Clean up the mask with morphology operations
        cleaned_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8))

        # Find contours in the mask
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Use the largest contour as the tracked object
            largest_contour = max(contours, key=cv2.contourArea)
            box_x, box_y, box_width, box_height = cv2.boundingRect(largest_contour)

        # Draw bounding box on a copy of the frame
        annotated_frame = frame.copy()
        cv2.rectangle(
            annotated_frame,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (0, 255, 0),
            2
        )

        processed_frames.append(annotated_frame)
        last_estimate = np.array([box_x, box_y, box_width, box_height, 0, 0])

        # Store bounding box coordinates in tracking results (1-indexed)
        bounding_box_log[str(frame_index + 1)] = [box_x, box_y, box_width, box_height]

    # Save output video
    save_video(FINAL_OUTPUT_PATH, processed_frames, fps, (frame_width, frame_height), is_color=True)
    logger.info(f"Tracked video saved to {FINAL_OUTPUT_PATH}")

    # Save tracking results as JSON
    with open(TRACKING_JSON_PATH, 'w') as tracking_file:
        json.dump(bounding_box_log, tracking_file, indent=2)
    logger.info(f"Tracking results saved to {TRACKING_JSON_PATH}")
    logger.info("Video Tracking - Completed Successfully")
