import numpy as np
import cv2
import json
from tqdm import tqdm

from paths import FINAL_OUTPUT_PATH, BACKGROUND_IMAGE_PATH, TRACKING_JSON_PATH
from utils import open_video, load_video_frames, save_video
from logger import get_logger

logger = get_logger()

def track_video(input_video_path):
    logger.info("Starting Tracking")

    cap, width, height, fps = open_video(input_video_path)
    frames = load_video_frames(cap, color_space='bgr')

    # Load and resize background image
    background = cv2.imread(BACKGROUND_IMAGE_PATH)
    if background is None:
        raise FileNotFoundError(f"Background image not found at {BACKGROUND_IMAGE_PATH}")
    background = cv2.resize(background, (width, height))

    tracked_frames = []
    tracking_results = {}

    x, y, w, h = width // 2, height // 2, 100, 200  # Default bbox
    for t, frame in enumerate(tqdm(frames, desc="Tracking: Processing frames")):
        diff = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(axis=2)
        mask = (diff > 30).astype(np.uint8)

        # Clean mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8))

        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        tracked_frames.append(frame)
        tracking_results[str(t + 1)] = [y, x, h, w]

    save_video(FINAL_OUTPUT_PATH, tracked_frames, fps, (width, height), is_color=True)
    logger.info(f"Finished writing tracking video to {FINAL_OUTPUT_PATH}")

    with open(TRACKING_JSON_PATH, 'w') as f:
        json.dump(tracking_results, f, indent=2)
    logger.info(f"Tracking data saved to {TRACKING_JSON_PATH}")
    print('************ Tracking COMPLETED! ************')
