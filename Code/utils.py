import numpy as np
import cv2
from scipy.stats import gaussian_kde


def memoize_dict(cache, key, compute_fn):
    """
    Return cached value if key exists; otherwise compute, store, and return it.

    Args:
        cache (dict): Dictionary to store computed values.
        key (hashable): Key to look up.
        compute_fn (callable): Function to compute value if not cached.

    Returns:
        Computed or cached value.
    """
    if key in cache:
        return cache[key]
    else:
        cache[key] = compute_fn(np.asarray(key))[0]
        return cache[key]


def scale_frame_border(frame):
    """
    Scale frame by 4% around the center to fix border issues.

    Args:
        frame (np.ndarray): Input image frame.

    Returns:
        np.ndarray: Scaled frame.
    """
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1.04)
    scaled_frame = cv2.warpAffine(frame, matrix, (width, height))
    return scaled_frame


def open_video(path):
    """
    Open video capture and get basic properties.

    Args:
        path (str): Path to video file.

    Returns:
        tuple: (cv2.VideoCapture, width, height, fps)
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height, fps


def close_video(cap):
    """
    Release video capture and close all OpenCV windows.

    Args:
        cap (cv2.VideoCapture): Video capture object.
    """
    cap.release()
    cv2.destroyAllWindows()


def moving_average(data, radius):
    """
    Smooth 1D data using moving average with reflection padding.

    Args:
        data (np.ndarray): 1D input array.
        radius (int): Radius of smoothing window.

    Returns:
        np.ndarray: Smoothed array.
    """
    window_size = 2 * radius + 1
    filter_kernel = np.ones(window_size) / window_size
    padded_data = np.pad(data, (radius, radius), 'reflect')

    # Fix padding at edges
    for i in range(radius):
        padded_data[i] = padded_data[radius] - padded_data[i]
    for i in range(len(padded_data) - 1, len(padded_data) - 1 - radius, -1):
        padded_data[i] = padded_data[len(padded_data) - radius - 1] - padded_data[i]

    smoothed = np.convolve(padded_data, filter_kernel, mode='same')
    return smoothed[radius:-radius]


def smooth_trajectory(trajectory, radius):
    """
    Smooth each dimension of a multi-dimensional trajectory array.

    Args:
        trajectory (np.ndarray): Array with shape (frames, dims).
        radius (int): Smoothing radius.

    Returns:
        np.ndarray: Smoothed trajectory.
    """
    smoothed = np.copy(trajectory)
    for dim in range(smoothed.shape[1]):
        smoothed[:, dim] = moving_average(trajectory[:, dim], radius=radius)
    return smoothed


def save_video(path, frames, fps, size, is_color):
    """
    Save a sequence of frames as a video file.

    Args:
        path (str): Output video path.
        frames (list or np.ndarray): Frames to write.
        fps (float): Frames per second.
        size (tuple): (width, height) of output video.
        is_color (bool): Whether frames are color or grayscale.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(path, fourcc, fps, size, isColor=is_color)
    for frame in frames:
        writer.write(frame)
    writer.release()


def normalize_to_255(matrix):
    """
    Normalize input matrix values to range 0-255 as uint8.

    Args:
        matrix (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized uint8 array.
    """
    if matrix.dtype == np.bool_:
        matrix = matrix.astype(np.uint8)
    matrix = matrix.astype(np.uint8)
    scaled = 255 * (matrix - np.min(matrix)) / np.ptp(matrix)
    return scaled.astype(np.uint8)


def load_video_frames(cap, color_space='bgr'):
    """
    Load all frames from a video capture and convert color space if needed.

    Args:
        cap (cv2.VideoCapture): Video capture object.
        color_space (str): 'bgr', 'yuv', 'bw', or 'hsv'.

    Returns:
        np.ndarray: Array of frames.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(n_frames):
        success, frame = cap.read()
        if not success:
            break
        if color_space == 'bgr':
            frames.append(frame)
        elif color_space == 'yuv':
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV))
        elif color_space == 'bw':
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames)


def apply_mask_to_frame(frame, mask):
    """
    Apply binary mask to each channel of a color frame.

    Args:
        frame (np.ndarray): Color image frame.
        mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Masked color frame.
    """
    masked_frame = np.copy(frame)
    for channel in range(3):
        masked_frame[:, :, channel] *= mask
    return masked_frame


def select_foreground_pixels(mask, count):
    """
    Randomly select indices of pixels where mask == 1.

    Args:
        mask (np.ndarray): Binary mask.
        count (int): Number of pixels to select.

    Returns:
        np.ndarray: Array of (row, col) indices.
    """
    indices = np.where(mask == 1)
    if len(indices[0]) == 0:
        return np.column_stack(indices)
    choices = np.random.choice(len(indices[0]), count)
    return np.column_stack((indices[0][choices], indices[1][choices]))


def select_background_pixels(mask, count):
    """
    Randomly select indices of pixels where mask == 0.

    Args:
        mask (np.ndarray): Binary mask.
        count (int): Number of pixels to select.

    Returns:
        np.ndarray: Array of (row, col) indices.
    """
    indices = np.where(mask == 0)
    if len(indices[0]) == 0:
        return np.column_stack(indices)
    choices = np.random.choice(len(indices[0]), count)
    return np.column_stack((indices[0][choices], indices[1][choices]))


def create_kde(data_points, bandwidth):
    """
    Create a KDE function based on provided data and bandwidth.

    Args:
        data_points (np.ndarray): Data points (N x D).
        bandwidth (float or str): Bandwidth parameter for KDE.

    Returns:
        callable: Function accepting points x, returning KDE evaluated values.
    """
    kde = gaussian_kde(data_points.T, bw_method=bandwidth)
    return lambda x: kde(x.T)


def create_disk_kernel(diameter):
    """
    Create an elliptical (disk-shaped) structuring element for morphology.

    Args:
        diameter (int): Diameter of the disk.

    Returns:
        np.ndarray: Structuring element.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
