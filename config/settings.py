from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

STORAGE_DIR = BASE_DIR / "storage"
DATABASE_DIR = STORAGE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "missing_child_system.db"
CHILD_IMAGE_DIR = STORAGE_DIR / "child_images"
TEMP_DIR = STORAGE_DIR / "temp"

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

MAX_UPLOAD_SIZE_MB = 5
MAX_IMAGES_PER_CHILD = 5
MIN_IMAGE_WIDTH = 80
MIN_IMAGE_HEIGHT = 80

FACE_EMBEDDING_BACKEND = "opencv_sface"

DEEPFACE_MODEL_NAME = "Facenet512"
DEEPFACE_DETECTOR_BACKEND = "opencv"

MODEL_DIR = STORAGE_DIR / "models"
OPENCV_YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
OPENCV_SFACE_MODEL_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
OPENCV_YUNET_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
OPENCV_SFACE_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
OPENCV_FACE_SCORE_THRESHOLD = 0.80
OPENCV_FACE_NMS_THRESHOLD = 0.30
OPENCV_FACE_TOP_K = 5000

MIN_FACE_IMAGE_QUALITY_SCORE = 35.0

SQLITE_TIMEOUT_SECONDS = 30
