import numpy as np
import cv2
from tqdm import tqdm

from utils import (
    create_disk_kernel,
    select_foreground_pixels,
    select_background_pixels,
    memoize_dict,
    create_kde,
    normalize_to_255,
    apply_mask_to_frame,
    open_video,
    load_video_frames,
    close_video,
    save_video
)
from paths import EXTRACTED_VIDEO_PATH, BINARY_MASK_VIDEO_PATH
from logger import get_logger

logger = get_logger()

# Constants controlling bandwidth for KDE smoothing
BW_MEDIUM = 1
BW_NARROW = 0.1

# Constants representing approximate pixel heights in the frame for body parts
LEGS_HEIGHT = 805
SHOES_HEIGHT = 870
SHOULDERS_HEIGHT = 405

# Threshold for blue channel masking (to isolate non-blue regions)
BLUE_MASK_THR = 140

# Window sizes for local regions around person or face
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 1000
FACE_WINDOW_HEIGHT = 250
FACE_WINDOW_WIDTH = 300


def study_frames_history(n_frames, frames_hsv, h, w):
    """
    Perform background subtraction on a sequence of HSV frames using KNN background subtractor.

    Args:
        n_frames (int): Number of frames to process.
        frames_hsv (list[np.ndarray]): List of frames in HSV color space.
        h (int): Frame height in pixels.
        w (int): Frame width in pixels.

    Returns:
        np.ndarray: A 3D numpy array of shape (n_frames, h, w) containing binary foreground masks per frame.
                    Masks are uint8 arrays with 1 indicating foreground pixels.
    """
    backSub = cv2.createBackgroundSubtractorKNN()
    mask_list = np.zeros((n_frames, h, w), dtype=np.uint8)
    logger.debug(f"Background Subtraction - BackgroundSubtractorKNN Studying Frames history")

    for j in tqdm(range(8), desc="[BS] - Passes"):
        logger.debug(f"Background Subtraction - BackgroundSubtractorKNN {j + 1} / 8 pass")
        for index_frame, frame in enumerate(tqdm(frames_hsv, desc=f"[BS] - Pass {j + 1}", leave=False)):
            # Use saturation and value channels only (skip hue)
            frame_sv = frame[:, :, 1:]
            fgMask = backSub.apply(frame_sv)
            fgMask = (fgMask > 200).astype(np.uint8)
            mask_list[index_frame] = fgMask

    logger.debug(f"Background Subtraction - BackgroundSubtractorKNN Finished")
    return mask_list


def collect_colors_body_and_shoes_kde(n_frames, frames_bgr, h, w, mask_list):
    """
    Collect color samples from foreground and background regions of the body and shoes for KDE modeling.

    Args:
        n_frames (int): Number of frames.
        frames_bgr (list[np.ndarray]): List of frames in BGR color space.
        h (int): Frame height.
        w (int): Frame width.
        mask_list (np.ndarray): Foreground masks from background subtraction (shape: n_frames x h x w).

    Returns:
        Tuple containing:
            - omega_f_colors (np.ndarray): Concatenated foreground body colors.
            - omega_b_colors (np.ndarray): Concatenated background body colors.
            - omega_f_shoes_colors (np.ndarray): Concatenated foreground shoe colors.
            - omega_b_shoes_colors (np.ndarray): Concatenated background shoe colors.
            - person_and_blue_mask_list (np.ndarray): List of masks combining person and blue-channel threshold.
    """
    omega_f_colors, omega_b_colors = None, None
    omega_f_shoes_colors, omega_b_shoes_colors = None, None
    person_and_blue_mask_list = np.zeros((n_frames, h, w))
    for frame_index, frame in enumerate(
            tqdm(frames_bgr, desc="Background Subtraction - Collecting body & shoes colors")):
        blue_frame, _, _ = cv2.split(frame)
        mask_for_frame = mask_list[frame_index].astype(np.uint8)
        mask_for_frame = cv2.morphologyEx(mask_for_frame, cv2.MORPH_CLOSE, create_disk_kernel(6))
        mask_for_frame = cv2.medianBlur(mask_for_frame, 7)
        contours, _ = cv2.findContours(mask_for_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        person_mask = np.zeros(mask_for_frame.shape)
        cv2.fillPoly(person_mask, pts=[contours[0]], color=1)
        blue_mask = (blue_frame < BLUE_MASK_THR).astype(np.uint8)
        person_and_blue_mask = (person_mask * blue_mask).astype(np.uint8)
        omega_f_indices = select_foreground_pixels(person_and_blue_mask, 20)
        omega_b_indices = select_background_pixels(person_and_blue_mask, 20)
        # Collect colors for shoes
        shoes_mask = np.copy(person_and_blue_mask)
        shoes_mask[:SHOES_HEIGHT, :] = 0
        omega_f_shoes_indices = select_foreground_pixels(shoes_mask, 20)
        shoes_mask = np.copy(person_and_blue_mask)
        shoes_mask[:SHOES_HEIGHT - 120, :] = 1
        omega_b_shoes_indices = select_background_pixels(shoes_mask, 20)
        person_and_blue_mask_list[frame_index] = person_and_blue_mask
        if omega_f_colors is None:
            omega_f_colors = frame[omega_f_indices[:, 0], omega_f_indices[:, 1], :]
            omega_b_colors = frame[omega_b_indices[:, 0], omega_b_indices[:, 1], :]
            omega_f_shoes_colors = frame[omega_f_shoes_indices[:, 0], omega_f_shoes_indices[:, 1], :]
            omega_b_shoes_colors = frame[omega_b_shoes_indices[:, 0], omega_b_shoes_indices[:, 1], :]
        else:
            omega_f_colors = np.concatenate((omega_f_colors, frame[omega_f_indices[:, 0], omega_f_indices[:, 1], :]))
            omega_b_colors = np.concatenate((omega_b_colors, frame[omega_b_indices[:, 0], omega_b_indices[:, 1], :]))
            omega_f_shoes_colors = np.concatenate(
                (omega_f_shoes_colors, frame[omega_f_shoes_indices[:, 0], omega_f_shoes_indices[:, 1], :]))
            omega_b_shoes_colors = np.concatenate(
                (omega_b_shoes_colors, frame[omega_b_shoes_indices[:, 0], omega_b_shoes_indices[:, 1], :]))

    return omega_f_colors, omega_b_colors, omega_f_shoes_colors, omega_b_shoes_colors, person_and_blue_mask_list


def filtering_body_and_shoes_kde(n_frames, frames_bgr, h, w, omega_f_colors, omega_b_colors, omega_f_shoes_colors,
                                 omega_b_shoes_colors, person_and_blue_mask_list):
    """
    Filter frames using KDEs built on collected colors for body and shoes to generate refined masks.

    Args:
        n_frames (int): Number of frames.
        frames_bgr (list[np.ndarray]): List of frames in BGR color space.
        h (int): Frame height.
        w (int): Frame width.
        omega_f_colors (np.ndarray): Foreground body color samples.
        omega_b_colors (np.ndarray): Background body color samples.
        omega_f_shoes_colors (np.ndarray): Foreground shoe color samples.
        omega_b_shoes_colors (np.ndarray): Background shoe color samples.
        person_and_blue_mask_list (np.ndarray): Masks combining person and blue threshold per frame.

    Returns:
        np.ndarray: Refined masks list (shape: n_frames x h x w) after KDE filtering.
    """
    foreground_pdf = create_kde(data_points=omega_f_colors, bandwidth=BW_MEDIUM)
    background_pdf = create_kde(data_points=omega_b_colors, bandwidth=BW_MEDIUM)
    foreground_shoes_pdf = create_kde(data_points=omega_f_shoes_colors, bandwidth=BW_MEDIUM)
    background_shoes_pdf = create_kde(data_points=omega_b_shoes_colors, bandwidth=BW_MEDIUM)

    foreground_pdf_memoization, background_pdf_memoization = dict(), dict()
    foreground_shoes_pdf_memoization, background_shoes_pdf_memoization = dict(), dict()
    or_mask_list = np.zeros((n_frames, h, w))
    # Filtering with KDEs general body parts & shoes
    for frame_index, frame in enumerate(tqdm(frames_bgr, desc="Background Subtraction - Filtering body & shoes KDE")):
        person_and_blue_mask = person_and_blue_mask_list[frame_index]
        person_and_blue_mask_indices = np.where(person_and_blue_mask == 1)
        y_mean, x_mean = int(np.mean(person_and_blue_mask_indices[0])), int(np.mean(person_and_blue_mask_indices[1]))
        small_frame_bgr = frame[max(0, y_mean - WINDOW_HEIGHT // 2):min(h, y_mean + WINDOW_HEIGHT // 2),
                          max(0, x_mean - WINDOW_WIDTH // 2):min(w, x_mean + WINDOW_WIDTH // 2),
                          :]
        small_person_and_blue_mask = person_and_blue_mask[
                                     max(0, y_mean - WINDOW_HEIGHT // 2):min(h, y_mean + WINDOW_HEIGHT // 2),
                                     max(0, x_mean - WINDOW_WIDTH // 2):min(w, x_mean + WINDOW_WIDTH // 2)]

        small_person_and_blue_mask_indices = np.where(small_person_and_blue_mask == 1)
        small_probs_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
        small_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: memoize_dict(foreground_pdf_memoization, elem, foreground_pdf),
                map(tuple, small_frame_bgr[small_person_and_blue_mask_indices])), dtype=float)
        small_background_probabilities_stacked = np.fromiter(
            map(lambda elem: memoize_dict(background_pdf_memoization, elem, background_pdf),
                map(tuple, small_frame_bgr[small_person_and_blue_mask_indices])), dtype=float)
        small_probs_fg_bigger_bg_mask[small_person_and_blue_mask_indices] = (
                small_foreground_probabilities_stacked > small_background_probabilities_stacked).astype(np.uint8)

        # Shoes restoration
        smaller_upper_white_mask = np.copy(small_probs_fg_bigger_bg_mask)
        smaller_upper_white_mask[:-270, :] = 1
        small_probs_fg_bigger_bg_mask_black_indices = np.where(smaller_upper_white_mask == 0)
        small_probs_shoes_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
        small_shoes_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: memoize_dict(foreground_shoes_pdf_memoization, elem, foreground_shoes_pdf),
                map(tuple, small_frame_bgr[small_probs_fg_bigger_bg_mask_black_indices])), dtype=float)

        small_shoes_background_probabilities_stacked = np.fromiter(
            map(lambda elem: memoize_dict(background_shoes_pdf_memoization, elem, background_shoes_pdf),
                map(tuple, small_frame_bgr[small_probs_fg_bigger_bg_mask_black_indices])), dtype=float)

        shoes_fg_shoes_bg_ratio = small_shoes_foreground_probabilities_stacked / (
                small_shoes_foreground_probabilities_stacked + small_shoes_background_probabilities_stacked)
        shoes_fg_beats_shoes_bg_mask = (shoes_fg_shoes_bg_ratio > 0.75).astype(np.uint8)
        small_probs_shoes_fg_bigger_bg_mask[small_probs_fg_bigger_bg_mask_black_indices] = shoes_fg_beats_shoes_bg_mask
        small_probs_shoes_fg_bigger_bg_mask_indices = np.where(small_probs_shoes_fg_bigger_bg_mask == 1)
        y_shoes_mean, x_shoes_mean = int(np.mean(small_probs_shoes_fg_bigger_bg_mask_indices[0])), int(
            np.mean(small_probs_shoes_fg_bigger_bg_mask_indices[1]))
        small_or_mask = np.zeros(small_probs_fg_bigger_bg_mask.shape)
        small_or_mask[:y_shoes_mean, :] = small_probs_fg_bigger_bg_mask[:y_shoes_mean, :]
        small_or_mask[y_shoes_mean:, :] = np.maximum(small_probs_fg_bigger_bg_mask[y_shoes_mean:, :],
                                                     small_probs_shoes_fg_bigger_bg_mask[y_shoes_mean:, :]).astype(
            np.uint8)

        DELTA_Y = 30
        small_or_mask[y_shoes_mean - DELTA_Y:, :] = cv2.morphologyEx(small_or_mask[y_shoes_mean - DELTA_Y:, :],
                                                                     cv2.MORPH_CLOSE, np.ones((1, 20)))
        small_or_mask[y_shoes_mean - DELTA_Y:, :] = cv2.morphologyEx(small_or_mask[y_shoes_mean - DELTA_Y:, :],
                                                                     cv2.MORPH_CLOSE, create_disk_kernel(20))

        or_mask = np.zeros(person_and_blue_mask.shape)
        or_mask[max(0, y_mean - WINDOW_HEIGHT // 2):min(h, y_mean + WINDOW_HEIGHT // 2),
        max(0, x_mean - WINDOW_WIDTH // 2):min(w, x_mean + WINDOW_WIDTH // 2)] = small_or_mask
        or_mask_list[frame_index] = or_mask
    return or_mask_list


def collect_colors_face_kde(frames_bgr, h, w, or_mask_list):
    """
    Collect color samples from face region foreground and background for KDE modeling.

    Args:
        n_frames (int): Number of frames.
        frames_bgr (list[np.ndarray]): List of frames in BGR color space.
        h (int): Frame height.
        w (int): Frame width.
        or_mask_list (np.ndarray): Refined masks from previous KDE filtering stage.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Foreground and background face color samples concatenated over all frames.
    """
    omega_f_face_colors, omega_b_face_colors = None, None
    # Collecting colors for building face KDE
    for frame_index, frame in enumerate(tqdm(frames_bgr, desc="Background Subtraction - Collecting face colors")):
        or_mask = or_mask_list[frame_index]
        face_mask = np.copy(or_mask)
        face_mask[SHOULDERS_HEIGHT:, :] = 0
        face_mask_indices = np.where(face_mask == 1)
        y_mean, x_mean = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))

        small_frame_bgr = frame[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2),
                          :]
        face_mask = np.copy(or_mask)  # Restore face mask again
        small_face_mask = face_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2)]

        small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, np.ones((20, 1), np.uint8))
        small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, np.ones((1, 20), np.uint8))

        omega_f_face_indices = select_foreground_pixels(small_face_mask, 20)
        omega_b_face_indices = select_background_pixels(small_face_mask, 20)
        if omega_f_face_colors is None:
            omega_f_face_colors = small_frame_bgr[omega_f_face_indices[:, 0], omega_f_face_indices[:, 1], :]
            omega_b_face_colors = small_frame_bgr[omega_b_face_indices[:, 0], omega_b_face_indices[:, 1], :]
        else:
            omega_f_face_colors = np.concatenate(
                (omega_f_face_colors, small_frame_bgr[omega_f_face_indices[:, 0], omega_f_face_indices[:, 1], :]))
            omega_b_face_colors = np.concatenate(
                (omega_b_face_colors, small_frame_bgr[omega_b_face_indices[:, 0], omega_b_face_indices[:, 1], :]))
    return omega_f_face_colors, omega_b_face_colors


def filtering_face_kde(frames_bgr, h, w, omega_f_face_colors, omega_b_face_colors, or_mask_list):
    """
    Apply KDE filtering on face region to further refine segmentation masks.

    Args:
        n_frames (int): Number of frames.
        frames_bgr (list[np.ndarray]): List of frames in BGR color space.
        h (int): Frame height.
        w (int): Frame width.
        omega_f_face_colors (np.ndarray): Foreground face color samples.
        omega_b_face_colors (np.ndarray): Background face color samples.
        or_mask_list (np.ndarray): Masks from previous filtering stage.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Lists of final binary masks and masked color frames for all frames.
    """
    foreground_face_pdf = create_kde(data_points=omega_f_face_colors, bandwidth=BW_NARROW)
    background_face_pdf = create_kde(data_points=omega_b_face_colors, bandwidth=BW_NARROW)
    foreground_face_pdf_memoization, background_face_pdf_memoization = dict(), dict()
    final_masks_list, final_frames_list = [], []
    # Final Processing of BS (applying face KDE)
    for frame_index, frame in enumerate(tqdm(frames_bgr, desc="Background Subtraction - Filtering face KDE")):
        or_mask = or_mask_list[frame_index]
        face_mask = np.copy(or_mask)
        face_mask[SHOULDERS_HEIGHT:, :] = 0
        face_mask_indices = np.where(face_mask == 1)
        y_mean, x_mean = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))

        small_frame_bgr = frame[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2),
                          :]
        face_mask = np.copy(or_mask)  # Restore face mask again
        small_face_mask = face_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2)]

        small_frame_bgr_stacked = small_frame_bgr.reshape((-1, 3))

        small_face_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: memoize_dict(foreground_face_pdf_memoization, elem, foreground_face_pdf),
                map(tuple, small_frame_bgr_stacked)), dtype=float)
        small_face_background_probabilities_stacked = np.fromiter(
            map(lambda elem: memoize_dict(background_face_pdf_memoization, elem, background_face_pdf),
                map(tuple, small_frame_bgr_stacked)), dtype=float)

        small_face_foreground_probabilities = small_face_foreground_probabilities_stacked.reshape(small_face_mask.shape)
        small_face_background_probabilities = small_face_background_probabilities_stacked.reshape(small_face_mask.shape)
        small_probs_face_fg_bigger_face_bg_mask = (
                small_face_foreground_probabilities > small_face_background_probabilities).astype(np.uint8)
        small_probs_face_fg_bigger_face_bg_mask_laplacian = cv2.Laplacian(small_probs_face_fg_bigger_face_bg_mask,
                                                                          cv2.CV_32F)
        small_probs_face_fg_bigger_face_bg_mask_laplacian = np.abs(small_probs_face_fg_bigger_face_bg_mask_laplacian)
        small_probs_face_fg_bigger_face_bg_mask = np.maximum(
            small_probs_face_fg_bigger_face_bg_mask - small_probs_face_fg_bigger_face_bg_mask_laplacian, 0)
        small_probs_face_fg_bigger_face_bg_mask[np.where(small_probs_face_fg_bigger_face_bg_mask > 1)] = 0
        small_probs_face_fg_bigger_face_bg_mask = small_probs_face_fg_bigger_face_bg_mask.astype(np.uint8)

        contours, _ = cv2.findContours(small_probs_face_fg_bigger_face_bg_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        small_contour_mask = np.zeros(small_probs_face_fg_bigger_face_bg_mask.shape, dtype=np.uint8)
        cv2.fillPoly(small_contour_mask, pts=[contours[0]], color=1)

        small_contour_mask = cv2.morphologyEx(small_contour_mask, cv2.MORPH_CLOSE, create_disk_kernel(12))
        small_contour_mask = cv2.dilate(small_contour_mask, create_disk_kernel(3), iterations=1).astype(np.uint8)
        small_contour_mask[-50:, :] = small_face_mask[-50:, :]

        final_mask = np.copy(or_mask).astype(np.uint8)
        final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
        max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2)] = small_contour_mask

        final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :] = cv2.morphologyEx(
            final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :], cv2.MORPH_OPEN,
            np.ones((6, 1), np.uint8))
        final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :] = cv2.morphologyEx(
            final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :], cv2.MORPH_OPEN,
            np.ones((1, 6), np.uint8))

        contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        final_contour_mask = np.zeros(final_mask.shape)
        cv2.fillPoly(final_contour_mask, pts=[contours[0]], color=1)
        final_mask = (final_contour_mask * final_mask).astype(np.uint8)
        final_masks_list.append(normalize_to_255(final_mask))
        final_frames_list.append(apply_mask_to_frame(frame=frame, mask=final_mask))
    return final_masks_list, final_frames_list


def write_final_video(final_masks_list, final_frames_list, cap, w, h, fps):
    """
    Save the final masked frames and binary masks as videos and close the video capture.

    Args:
        final_masks_list (list[np.ndarray]): List of final binary masks.
        final_frames_list (list[np.ndarray]): List of masked color frames.
        cap (cv2.VideoCapture): OpenCV video capture object to close.
        w (int): Frame width.
        h (int): Frame height.
        fps (float): Frames per second of the video.
    """
    save_video(path=EXTRACTED_VIDEO_PATH, frames=final_frames_list, fps=fps, size=(w, h), is_color=True)
    logger.info(f"Extracted (foreground) video saved to {EXTRACTED_VIDEO_PATH}")

    save_video(path=BINARY_MASK_VIDEO_PATH, frames=final_masks_list, fps=fps, size=(w, h), is_color=False)
    logger.info(f"Binary mask video saved to {BINARY_MASK_VIDEO_PATH}")

    logger.debug('Background Subtraction - Completed Successfully')

    close_video(cap)


def background_subtraction(input_video_path):
    """
    Main function to run the background subtraction pipeline on an input video.

    Steps:
      - Open video and load frames in BGR and HSV color spaces.
      - Perform background subtraction to get initial masks.
      - Collect color samples for body and shoes KDE modeling.
      - Filter masks with KDEs for body and shoes.
      - Collect face color samples and filter masks with face KDE.
      - Write the final masked video and binary masks to disk.

    Args:
        input_video_path (str): Path to the input video file.
    """
    logger.info('Background Subtraction - Starting Process')
    # Read input video
    cap, w, h, fps = open_video(path=input_video_path)
    # Get frame count and frames in two color spaces
    frames_bgr = load_video_frames(cap, color_space='bgr')
    frames_hsv = load_video_frames(cap, color_space='hsv')
    n_frames = len(frames_bgr)
    mask_list = study_frames_history(n_frames, frames_hsv, h, w)
    omega_f_colors, omega_b_colors, omega_f_shoes_colors, omega_b_shoes_colors, person_and_blue_mask_list = collect_colors_body_and_shoes_kde(
        n_frames, frames_bgr, h, w, mask_list)
    or_mask_list = filtering_body_and_shoes_kde(n_frames, frames_bgr, h, w, omega_f_colors, omega_b_colors,
                                                omega_f_shoes_colors, omega_b_shoes_colors, person_and_blue_mask_list)
    omega_f_face_colors, omega_b_face_colors = collect_colors_face_kde(frames_bgr, h, w, or_mask_list)
    final_masks_list, final_frames_list = filtering_face_kde(frames_bgr, h, w, omega_f_face_colors,
                                                             omega_b_face_colors, or_mask_list)
    write_final_video(final_masks_list, final_frames_list, cap, w, h, fps)
