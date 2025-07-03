import os

# Base directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define project-level folders (relative to the project root)
INPUTS_DIR = os.path.join(BASE_DIR, '..', 'Inputs')
OUTPUTS_DIR = os.path.join(BASE_DIR, '..', 'Outputs')
TEMP_DIR = os.path.join(BASE_DIR, '..', 'Temp')
DOCUMENT_DIR = os.path.join(BASE_DIR, '..', 'Document')
SCREENREC_DIR = os.path.join(BASE_DIR, '..', 'ScreenRec')

# Ensure these directories exist
for folder in [OUTPUTS_DIR, TEMP_DIR, DOCUMENT_DIR, SCREENREC_DIR]:
    os.makedirs(folder, exist_ok=True)

# Input files
INPUT_VIDEO_PATH = os.path.join(INPUTS_DIR, 'INPUT.avi')
BACKGROUND_IMAGE_PATH = os.path.join(INPUTS_DIR, 'background.jpg')

# Output files
STABILIZED_VIDEO_PATH = os.path.join(OUTPUTS_DIR, 'stabilize.avi')
BINARY_MASK_VIDEO_PATH = os.path.join(OUTPUTS_DIR, 'binary.avi')
MATTED_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, 'matted.avi')
ALPHA_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, 'alpha.avi')
LOG_PATH = os.path.join(OUTPUTS_DIR, 'log.txt')

# Temp files
TRANSFORMS_PATH = os.path.join(TEMP_DIR, 'transforms_video_stab.np')

# You can define specific files under Documents or ScreenRec like:
DOCUMENT_REPORT_PATH = os.path.join(DOCUMENT_DIR, 'report.txt')
SCREEN_RECORDING_PATH = os.path.join(SCREENREC_DIR, 'recording.mp4')
