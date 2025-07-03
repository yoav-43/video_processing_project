import numpy as np
import cv2
from utils import get_video_files, load_entire_video, write_video
import logging


my_logger = logging.getLogger('MyLogger')


def track_video(input_video_path):
    my_logger.info('Starting Tracking')

    cap_stabilize, video_width, video_height, fps = get_video_files(path=input_video_path)
    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')
    font, bottom_left_corner_of_text, font_scale, font_color, line_type = cv2.FONT_HERSHEY_SIMPLEX, (50, 50), 1, (
    0, 0, 255), 2

    instruction_frame = cv2.putText(frames_bgr[0],
                                    "Select a Rectangle and then press SPACE or ENTER button! or 'ESC' key for auto selection.",
                                    bottom_left_corner_of_text,
                                    font,
                                    font_scale,
                                    font_color,
                                    line_type)
    initBB = cv2.selectROI("Frame", instruction_frame, fromCenter=False, showCrosshair=True)

    x, y, w, h = initBB
    if any([x == 0, y == 0, w == 0, h == 0]):
        x, y, w, h = 180, 60, 340, 740  # simply hardcoded the values
    track_window = (x, y, w, h)

    # set up the ROI for tracking
    roi = frames_bgr[0][y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    reducing_light_mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], reducing_light_mask, [256, 256], [0, 256, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # Setup the termination criteria, 10 iterations
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0)
    tracking_frames_list = [cv2.rectangle(frames_bgr[0], (x, y), (x + w, y + h), (0, 255, 0), 2)]
    for frame_index, frame in enumerate(frames_bgr[1:]):
        print(f"[Tracking] Using MeanShift - Frame: {frame_index} / {len(frames_bgr) - 1}")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 256, 0, 256], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        tracked_img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        tracking_frames_list.append(tracked_img)

    write_video('../Outputs/OUTPUT.avi', tracking_frames_list, fps, (video_width, video_height), is_color=True)
    print('~~~~~~~~~~~ [Tracking] FINISHED! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ OUTPUT.avi has been created! ~~~~~~~~~~~')
    my_logger.info('Finished Tracking')
