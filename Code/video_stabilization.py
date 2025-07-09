import numpy as np
import cv2

from utils import (
    open_video,
    load_video_frames,
    close_video,
    save_video,
    smooth_trajectory,
    scale_frame_border
)
from paths import STABILIZED_VIDEO_PATH, TRANSFORMS_PATH
from tqdm import tqdm
from logger import get_logger
logger = get_logger()

MAX_CORNERS = 500
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 30
BLOCK_SIZE = 3
smooth_trajectory_RADIUS = 5


def stabilize_video(input_video_path):
    logger.info('Video Stabilization - Initializing process.')

    cap, w, h, fps = open_video(path=input_video_path)
    frames_bgr = load_video_frames(cap, color_space='bgr')

    n_frames = len(frames_bgr)
    prev = frames_bgr[0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames - 1, 9), np.float32)
    transforms_list = np.zeros((n_frames - 1, 3, 3), np.float32)

    for frame_index, curr in tqdm(enumerate(frames_bgr[1:]), total=n_frames - 1, desc="Estimating transforms"):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=MAX_CORNERS,
                                           qualityLevel=QUALITY_LEVEL,
                                           minDistance=MIN_DISTANCE,
                                           blockSize=BLOCK_SIZE)

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        idx = np.where(status == 1)[0]
        prev_pts, curr_pts = prev_pts[idx], curr_pts[idx]

        transform_matrix, _ = cv2.findHomography(prev_pts, curr_pts)
        transforms[frame_index] = transform_matrix.flatten()

        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smooth_trajectoryed_trajectory = smooth_trajectory(trajectory, smooth_trajectory_RADIUS)
    difference = smooth_trajectoryed_trajectory - trajectory
    transforms_smooth_trajectory = transforms + difference

    stabilized_frames_list = [frames_bgr[0]]

    for frame_index, frame in tqdm(enumerate(frames_bgr[:-1]), total=n_frames - 1, desc="Warping frames"):
        transform_matrix = transforms_smooth_trajectory[frame_index].reshape((3, 3))
        frame_stabilized = cv2.warpPerspective(frame, transform_matrix, (w, h))
        frame_stabilized = scale_frame_border(frame_stabilized)

        stabilized_frames_list.append(frame_stabilized)
        transforms_list[frame_index] = transform_matrix

    close_video(cap)

    save_video(STABILIZED_VIDEO_PATH, stabilized_frames_list, fps, (w, h), is_color=True)
    transforms_list.dump(TRANSFORMS_PATH)


    logger.info('Video Stabilization - completed successfully')
    logger.info('Video Stabilization - Output saved to stabilize.avi')
    logger.info('Video Stabilization - process finished')
