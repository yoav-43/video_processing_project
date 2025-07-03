import os

# Base directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Your student IDs
ID1 = "209619477"
ID2 = "209518299"
ID_SUFFIX = f"{ID1}_{ID2}"

# Define folders
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

# Output videos (must include ID suffix)
STABILIZED_VIDEO_PATH = os.path.join(OUTPUTS_DIR, f'stabilize_{ID_SUFFIX}.avi')
EXTRACTED_VIDEO_PATH = os.path.join(OUTPUTS_DIR, f'extracted_{ID_SUFFIX}.avi')
BINARY_MASK_VIDEO_PATH = os.path.join(OUTPUTS_DIR, f'binary_{ID_SUFFIX}.avi')
MATTED_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, f'matted_{ID_SUFFIX}.avi')
ALPHA_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, f'alpha_{ID_SUFFIX}.avi')
FINAL_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, f'OUTPUT_{ID_SUFFIX}.avi')
LOG_PATH = os.path.join(OUTPUTS_DIR, 'log.txt')

# JSON outputs
TIMING_JSON_PATH = os.path.join(OUTPUTS_DIR, 'timing.json')
TRACKING_JSON_PATH = os.path.join(OUTPUTS_DIR, 'tracking.json')

# Temp
TRANSFORMS_PATH = os.path.join(TEMP_DIR, 'transforms_video_stab.npy')

# Optional
DOCUMENT_REPORT_PATH = os.path.join(DOCUMENT_DIR, 'report.txt')
SCREEN_RECORDING_PATH = os.path.join(SCREENREC_DIR, 'recording.mp4')
