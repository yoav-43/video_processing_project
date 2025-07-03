import argparse
import sys

from video_stabilization import stabilize_video
from background_subtraction import background_subtraction
from matting import video_matting
from tracking import track_video
from logger import get_logger
from paths import (
    INPUT_VIDEO_PATH,
    STABILIZED_VIDEO_PATH,
    BINARY_MASK_VIDEO_PATH,
    BACKGROUND_IMAGE_PATH,
    MATTED_OUTPUT_PATH,
    ALPHA_OUTPUT_PATH
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pass no arguments or use -all to run the full pipeline. "
                    "Use individual flags to run specific parts.")
    parser.add_argument('-vs', action='store_true', help='Run video stabilization')
    parser.add_argument('-bg', action='store_true', help='Run background subtraction')
    parser.add_argument('-mt', action='store_true', help='Run video matting')
    parser.add_argument('-tk', action='store_true', help='Run tracking')
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger = get_logger()
    if len(sys.argv) == 1:
        args.vs = True
        args.bg = True
        args.mt = True
        args.tk = True

    if args.vs:
        stabilize_video(INPUT_VIDEO_PATH)

    if args.bg:
        background_subtraction(STABILIZED_VIDEO_PATH)

    if args.mt:
        video_matting(
            STABILIZED_VIDEO_PATH,
            BINARY_MASK_VIDEO_PATH,
            BACKGROUND_IMAGE_PATH,
            MATTED_OUTPUT_PATH,
            ALPHA_OUTPUT_PATH
        )

    if args.tk:
        track_video(MATTED_OUTPUT_PATH)


if __name__ == "__main__":
    main()