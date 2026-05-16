import cv2
import mediapipe as mp
import urllib.request
import time
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "models" / "holistic_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"

if not MODEL_PATH.exists():
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO
)

def draw_landmarks_safe(frame, landmark_list, color=(0, 255, 0)):
    """Draw landmarks on the frame while tolerating different input formats.

    Args:
        frame: Image frame where the points will be drawn.
        landmark_list: MediaPipe landmark list or an equivalent structure.
        color: Point color in BGR format.

    Returns:
        None.
    """
    if not landmark_list:
        return
        
    height, width = frame.shape[:2]
    
    if isinstance(landmark_list, list) and len(landmark_list) > 0 and isinstance(landmark_list[0], list):
        points_to_draw = landmark_list[0]
    elif hasattr(landmark_list, 'landmarks'):
        points_to_draw = landmark_list.landmarks
    elif isinstance(landmark_list, list):
        points_to_draw = landmark_list
    else:
        return

    for landmark in points_to_draw:
        try:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(frame, point, 2, color, -1)
        except AttributeError:
            continue

# webcam init
webcam = cv2.VideoCapture(0)
start_time = time.time()

# full screen
WINDOW_NAME = "Holistic Capture"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with HolisticLandmarker.create_from_options(options) as landmarker:
    while webcam.isOpened():
        success, frame = webcam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        timestamp_ms = int((time.time() - start_time) * 1000)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # draw face 
        if result.face_landmarks:
            draw_landmarks_safe(frame, result.face_landmarks, (255, 0, 0))

        # draw body
        if result.pose_landmarks:
            draw_landmarks_safe(frame, result.pose_landmarks, (0, 255, 0))

        # draw hands
        if result.left_hand_landmarks:
            draw_landmarks_safe(frame, result.left_hand_landmarks, (0, 0, 255))
        if result.right_hand_landmarks:
            draw_landmarks_safe(frame, result.right_hand_landmarks, (0, 0, 255))

        cv2.imshow(WINDOW_NAME, frame)

        # press ESC to close
        if cv2.waitKey(1) & 0xFF == 27:  
            break

webcam.release()
cv2.destroyAllWindows()