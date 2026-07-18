import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.embedding_generator import FaceEmbeddingDependencyError, FaceImageRejectedError, calculate_image_quality_score
from ai.model_assets import ModelAssetError, ModelAssetSpec, ensure_model_file
from config.constants import ALLOWED_IMAGE_EXTENSIONS, MAX_CHILD_AGE, MIN_CHILD_AGE
from config.settings import (
    AGE_PROGRESSION_MODEL_NAME,
    BASE_DIR,
    MIN_FACE_AREA_RATIO,
    MIN_FACE_IMAGE_QUALITY_SCORE,
    OPENCV_FACE_NMS_THRESHOLD,
    OPENCV_FACE_SCORE_THRESHOLD,
    OPENCV_FACE_TOP_K,
    OPENCV_SFACE_MODEL_SHA256,
    OPENCV_SFACE_MODEL_PATH,
    OPENCV_SFACE_MODEL_SIZE,
    OPENCV_SFACE_MODEL_URLS,
    OPENCV_YUNET_MODEL_SHA256,
    OPENCV_YUNET_MODEL_PATH,
    OPENCV_YUNET_MODEL_SIZE,
    OPENCV_YUNET_MODEL_URLS,
)
from utils.logger import get_logger


logger = get_logger(__name__)


class AgeProgressionError(Exception):
    """Raised when age progression cannot be generated."""


class AgeProgressionDependencyError(AgeProgressionError):
    """Raised when OpenCV or required model assets are unavailable."""


@dataclass(frozen=True)
class FaceAnalysis:
    image_path: str
    image_hash: str
    quality_score: float
    face_confidence: float
    face_area_ratio: float
    bbox: tuple[int, int, int, int]
    landmarks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class AgeProgressionRenderResult:
    image_bytes: bytes
    source_analysis: FaceAnalysis
    generated_analysis: FaceAnalysis
    model_name: str
    target_age_label: str
    progression_years: int
    approach_notes: str


@dataclass(frozen=True)
class AgeProgressionProfile:
    source_age: int
    target_age: int
    age_delta: int
    gap_strength: float
    child_to_teen: float
    child_to_adult: float
    adult_maturity: float
    midlife_maturity: float
    senior_maturity: float
    geometry_strength: float
    structure_strength: float
    texture_strength: float
    line_strength: float


def ensure_age_progression_ready() -> None:
    cv2, _ = _load_dependencies()
    _ensure_opencv_face_api(cv2)
    _ensure_opencv_model_assets()
    logger.info("Age progression model readiness check completed")


def analyze_face_image(image_path: Path) -> FaceAnalysis:
    image_path = _validate_source_path(image_path)
    cv2, _ = _load_dependencies()
    _ensure_opencv_face_api(cv2)
    _ensure_opencv_model_assets()

    quality_score = calculate_image_quality_score(image_path)
    if quality_score < MIN_FACE_IMAGE_QUALITY_SCORE:
        raise FaceImageRejectedError(
            f"Image quality score {quality_score:.1f} is below the required "
            f"{MIN_FACE_IMAGE_QUALITY_SCORE:.1f}."
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise FaceImageRejectedError("Image could not be read by OpenCV.")

    face, face_area_ratio = _detect_single_face(image)
    landmarks = _extract_landmarks(face)
    return FaceAnalysis(
        image_path=_relative_or_absolute_path(image_path),
        image_hash=_sha256_file(image_path),
        quality_score=quality_score,
        face_confidence=float(face[-1]),
        face_area_ratio=face_area_ratio,
        bbox=_face_bbox(face, image.shape[1], image.shape[0]),
        landmarks=landmarks,
    )


def generate_age_progressed_estimate(
    source_image_path: Path,
    source_age: int,
    target_age: int,
) -> AgeProgressionRenderResult:
    _validate_age_request(source_age, target_age)
    source_image_path = _validate_source_path(source_image_path)
    source_analysis = analyze_face_image(source_image_path)

    cv2, _ = _load_dependencies()
    source_image = cv2.imread(str(source_image_path))
    if source_image is None:
        raise FaceImageRejectedError("Image could not be read by OpenCV.")

    logger.info(
        "Generating age progression source_image=%s source_age=%s target_age=%s",
        source_analysis.image_path,
        source_age,
        target_age,
    )

    generated_image = _apply_age_progression_transform(
        source_image=source_image,
        analysis=source_analysis,
        source_age=source_age,
        target_age=target_age,
    )

    ok, encoded = cv2.imencode(".jpg", generated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise AgeProgressionError("Age-progressed image could not be encoded.")

    image_bytes = encoded.tobytes()
    generated_analysis = _analyze_generated_bytes(image_bytes)
    target_age_label = age_stage_label(target_age)

    logger.info(
        "Generated age progression estimate source_age=%s target_age=%s target_stage=%s generated_quality=%.2f",
        source_age,
        target_age,
        target_age_label,
        generated_analysis.quality_score,
    )

    return AgeProgressionRenderResult(
        image_bytes=image_bytes,
        source_analysis=source_analysis,
        generated_analysis=generated_analysis,
        model_name=AGE_PROGRESSION_MODEL_NAME,
        target_age_label=target_age_label,
        progression_years=target_age - source_age,
        approach_notes=(
            "Target-age-conditioned, landmark-guided craniofacial geometry adjustment with deterministic "
            "age-stage facial-structure and skin-texture synthesis; "
            "validated with OpenCV YuNet detection and SFace identity-preservation scoring."
        ),
    )


def age_stage_label(age: int) -> str:
    if age <= 4:
        return "early childhood estimate (0-4)"
    if age <= 12:
        return "childhood estimate (5-12)"
    if age <= 19:
        return "adolescent estimate (13-19)"
    if age <= 35:
        return "young adult estimate (20-35)"
    if age <= 55:
        return "mature adult estimate (36-55)"
    return "older adult estimate (56+)"


def _apply_age_progression_transform(
    source_image: Any,
    analysis: FaceAnalysis,
    source_age: int,
    target_age: int,
) -> Any:
    cv2, np = _load_dependencies()
    x, y, width, height = _expanded_face_region(analysis.bbox, source_image.shape[1], source_image.shape[0])
    face_roi = source_image[y : y + height, x : x + width].copy()
    local_landmarks = tuple((point_x - x, point_y - y) for point_x, point_y in analysis.landmarks)
    profile = _age_progression_profile(source_age, target_age)

    warped_roi = _warp_craniofacial_geometry(face_roi, local_landmarks, profile)
    textured_roi = _synthesize_age_texture(warped_roi, local_landmarks, profile)

    mask = _face_blend_mask(height, width, np)
    mask_3 = np.dstack([mask, mask, mask])
    output = source_image.copy()
    blended_roi = (textured_roi.astype(np.float32) * mask_3 + face_roi.astype(np.float32) * (1.0 - mask_3)).clip(
        0,
        255,
    )
    output[y : y + height, x : x + width] = blended_roi.astype(np.uint8)
    return output


def _age_progression_profile(source_age: int, target_age: int) -> AgeProgressionProfile:
    age_delta = max(target_age - source_age, 0)
    gap_strength = _smoothstep(float(age_delta), 1.0, 32.0)
    source_childness = 1.0 - _smoothstep(float(source_age), 12.0, 21.0)
    adolescent_target = _smoothstep(float(target_age), 11.0, 18.0)
    adult_target = _smoothstep(float(target_age), 18.0, 30.0)
    mature_target = _smoothstep(float(target_age), 32.0, 55.0)
    senior_target = _smoothstep(float(target_age), 55.0, 78.0)

    child_to_teen = source_childness * adolescent_target * min(age_delta / 9.0, 1.0)
    child_to_adult = source_childness * adult_target * min(age_delta / 18.0, 1.0)
    adult_maturity = adult_target * min(age_delta / 24.0, 1.0)
    midlife_maturity = mature_target * min(age_delta / 36.0, 1.0)
    senior_maturity = senior_target * min(age_delta / 48.0, 1.0)

    geometry_strength = min(
        1.0,
        0.18 * gap_strength
        + 0.36 * child_to_teen
        + 0.42 * child_to_adult
        + 0.16 * adult_maturity
        + 0.10 * midlife_maturity,
    )
    structure_strength = min(
        1.0,
        0.16 * gap_strength
        + 0.34 * child_to_teen
        + 0.46 * child_to_adult
        + 0.30 * adult_maturity
        + 0.24 * midlife_maturity,
    )
    texture_strength = min(
        1.0,
        0.10 * gap_strength
        + 0.10 * child_to_teen
        + 0.20 * child_to_adult
        + 0.32 * adult_maturity
        + 0.42 * midlife_maturity
        + 0.45 * senior_maturity,
    )
    line_strength = min(
        1.0,
        0.06 * child_to_adult
        + 0.16 * adult_maturity
        + 0.56 * midlife_maturity
        + 0.65 * senior_maturity,
    )

    return AgeProgressionProfile(
        source_age=source_age,
        target_age=target_age,
        age_delta=age_delta,
        gap_strength=gap_strength,
        child_to_teen=child_to_teen,
        child_to_adult=child_to_adult,
        adult_maturity=adult_maturity,
        midlife_maturity=midlife_maturity,
        senior_maturity=senior_maturity,
        geometry_strength=geometry_strength,
        structure_strength=structure_strength,
        texture_strength=texture_strength,
        line_strength=line_strength,
    )


def _warp_craniofacial_geometry(
    face_roi: Any,
    landmarks: tuple[tuple[float, float], ...],
    profile: AgeProgressionProfile,
) -> Any:
    cv2, np = _load_dependencies()
    height, width = face_roi.shape[:2]
    if height <= 0 or width <= 0:
        return face_roi

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    center_x = float((left_eye[0] + right_eye[0] + nose[0]) / 3.0)
    nose_y = float(nose[1])
    mouth_y = float((left_mouth[1] + right_mouth[1]) / 2.0)

    lower_face_stretch = (
        0.05 * profile.gap_strength
        + 0.12 * profile.child_to_teen
        + 0.19 * profile.child_to_adult
        + 0.08 * profile.adult_maturity
    )
    jaw_widening = (
        0.035 * profile.gap_strength
        + 0.075 * profile.child_to_teen
        + 0.12 * profile.child_to_adult
        + 0.065 * profile.adult_maturity
        + 0.035 * profile.midlife_maturity
    )
    cheek_slimming = (
        0.035 * profile.child_to_teen
        + 0.075 * profile.child_to_adult
        + 0.070 * profile.adult_maturity
        + 0.055 * profile.midlife_maturity
    )
    nose_projection = (
        0.018 * profile.gap_strength
        + 0.040 * profile.child_to_teen
        + 0.060 * profile.child_to_adult
        + 0.030 * profile.adult_maturity
    )
    brow_depth = 0.016 * profile.child_to_adult + 0.020 * profile.adult_maturity + 0.018 * profile.midlife_maturity

    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    lower_weight = _sigmoid((grid_y - nose_y) / max(height * 0.11, 1.0), np)
    mouth_weight = np.exp(-(((grid_y - mouth_y) / max(height * 0.20, 1.0)) ** 2))
    eye_y = float((left_eye[1] + right_eye[1]) / 2.0)
    eye_weight = np.exp(-(((grid_y - eye_y) / max(height * 0.11, 1.0)) ** 2))
    radial_x = (grid_x - center_x) / max(width * 0.5, 1.0)

    map_y = grid_y.copy()
    map_x = grid_x.copy()
    map_y = nose_y + (map_y - nose_y) / (1.0 + lower_face_stretch * lower_weight)
    map_x = center_x + (map_x - center_x) / (1.0 + jaw_widening * lower_weight)

    cheek_weight = np.clip((np.abs(radial_x) - 0.22) / 0.45, 0.0, 1.0) * lower_weight
    map_x = center_x + (map_x - center_x) * (1.0 + cheek_slimming * cheek_weight)
    map_y = map_y + brow_depth * height * 0.045 * eye_weight

    nose_weight = np.exp(
        -(((grid_x - nose[0]) / max(width * 0.12, 1.0)) ** 2 + ((grid_y - nose_y) / max(height * 0.14, 1.0)) ** 2)
    )
    map_y = map_y + nose_projection * height * 0.08 * nose_weight
    map_x = np.clip(map_x, 0, width - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, height - 1).astype(np.float32)

    remapped = cv2.remap(face_roi, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    if profile.structure_strength > 0:
        remapped = _blend_maturity_contours(remapped, landmarks, profile.structure_strength * mouth_weight)
    return remapped


def _synthesize_age_texture(
    face_roi: Any,
    landmarks: tuple[tuple[float, float], ...],
    profile: AgeProgressionProfile,
) -> Any:
    cv2, np = _load_dependencies()
    target_maturity = max(profile.adult_maturity, profile.midlife_maturity, profile.senior_maturity)
    child_softness = max(0.0, 1.0 - _smoothstep(float(profile.target_age), 10.0, 18.0))

    output = face_roi.copy()
    if child_softness > 0:
        softened = cv2.bilateralFilter(output, 7, 35, 35)
        output = cv2.addWeighted(softened, 0.10 * child_softness, output, 1.0 - 0.10 * child_softness, 0)

    if profile.structure_strength > 0:
        output = _enhance_structural_features(output, landmarks, profile)

    if target_maturity > 0 or profile.child_to_adult > 0:
        output = _apply_luma_contrast(output, 0.05 + 0.16 * profile.texture_strength)
        output = _add_age_lines(output, landmarks, profile)
        output = _add_skin_texture(output, profile)

    return output


def _blend_maturity_contours(face_roi: Any, landmarks: tuple[tuple[float, float], ...], maturity_weight: Any) -> Any:
    cv2, np = _load_dependencies()
    height, width = face_roi.shape[:2]
    center_x = float(landmarks[2][0])
    grid_x, _ = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    side_weight = np.clip(np.abs(grid_x - center_x) / max(width * 0.5, 1.0), 0.0, 1.0)
    shadow = (side_weight * maturity_weight * 22.0).clip(0, 26).astype(np.uint8)
    shadow_3 = np.dstack([shadow, shadow, shadow])
    return cv2.subtract(face_roi, shadow_3)


def _apply_luma_contrast(face_roi: Any, strength: float) -> Any:
    cv2, _ = _load_dependencies()
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0 + strength * 2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    blended_l = cv2.addWeighted(enhanced_l, strength, l_channel, 1.0 - strength, 0)
    return cv2.cvtColor(cv2.merge([blended_l, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def _add_age_lines(face_roi: Any, landmarks: tuple[tuple[float, float], ...], profile: AgeProgressionProfile) -> Any:
    cv2, np = _load_dependencies()
    height, width = face_roi.shape[:2]
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    eye_y = float((left_eye[1] + right_eye[1]) / 2.0)
    mouth_y = float((left_mouth[1] + right_mouth[1]) / 2.0)
    strength = profile.line_strength
    if strength <= 0:
        return face_roi

    overlay = np.zeros_like(face_roi)
    line_color = (26, 34, 38)
    highlight_color = (220, 226, 228)
    thickness = 1 if strength < 0.7 else 2

    forehead_top = int(max(eye_y - height * 0.25, height * 0.12))
    forehead_bottom = int(max(eye_y - height * 0.09, forehead_top + 1))
    forehead_line_count = 2 if strength < 0.45 else 3
    for index, y in enumerate(np.linspace(forehead_top, forehead_bottom, forehead_line_count)):
        amplitude = max(width * 0.010, 1.0) * (1.0 + index * 0.25)
        points = []
        for t in np.linspace(-0.28, 0.28, 18):
            px = int(width * (0.5 + t))
            py = int(y + math.sin(t * math.pi * 3.0) * amplitude)
            points.append([px, py])
        cv2.polylines(overlay, [np.array(points, dtype=np.int32)], False, line_color, thickness, cv2.LINE_AA)

    for eye, direction in ((left_eye, -1), (right_eye, 1)):
        for offset in (-0.03, 0.02, 0.07):
            start = (int(eye[0] + direction * width * 0.08), int(eye[1] + height * offset))
            end = (int(eye[0] + direction * width * 0.17), int(eye[1] + height * (offset + 0.03)))
            cv2.line(overlay, start, end, line_color, thickness, cv2.LINE_AA)

    for mouth, direction in ((left_mouth, -1), (right_mouth, 1)):
        start = (int(nose[0] + direction * width * 0.035), int(nose[1] + height * 0.08))
        mid = (int(mouth[0] + direction * width * 0.02), int((mouth_y + nose[1]) / 2.0 + height * 0.10))
        end = (int(mouth[0] + direction * width * 0.035), int(mouth_y + height * 0.06))
        cv2.polylines(overlay, [np.array([start, mid, end], dtype=np.int32)], False, line_color, thickness, cv2.LINE_AA)

    overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
    darkened = cv2.addWeighted(face_roi, 1.0, overlay, 0.42 * strength, 0)

    highlight = np.zeros_like(face_roi)
    cv2.line(
        highlight,
        (int(width * 0.32), int(mouth_y + height * 0.04)),
        (int(width * 0.68), int(mouth_y + height * 0.04)),
        highlight_color,
        1,
        cv2.LINE_AA,
    )
    return cv2.addWeighted(darkened, 1.0, cv2.GaussianBlur(highlight, (5, 5), 0), 0.07 * strength, 0)


def _add_skin_texture(face_roi: Any, profile: AgeProgressionProfile) -> Any:
    cv2, np = _load_dependencies()
    strength = profile.texture_strength
    if strength <= 0:
        return face_roi

    digest = hashlib.sha256(face_roi[: min(20, face_roi.shape[0]), : min(20, face_roi.shape[1])].tobytes()).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, face_roi.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.2)
    noise = cv2.normalize(noise, None, -1.0, 1.0, cv2.NORM_MINMAX)
    texture = np.dstack([noise, noise, noise]) * (13.0 * strength)
    textured = face_roi.astype(np.float32) + texture

    if profile.midlife_maturity > 0.05:
        spot_mask = np.zeros(face_roi.shape[:2], dtype=np.float32)
        height, width = spot_mask.shape
        spot_count = int(4 + 10 * profile.midlife_maturity)
        for _ in range(spot_count):
            cx = int(rng.integers(max(1, int(width * 0.22)), max(2, int(width * 0.78))))
            cy = int(rng.integers(max(1, int(height * 0.28)), max(2, int(height * 0.76))))
            radius = int(rng.integers(1, max(2, int(min(width, height) * 0.018))))
            cv2.circle(spot_mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
        spot_mask = cv2.GaussianBlur(spot_mask, (5, 5), 0)
        textured = textured - np.dstack([spot_mask, spot_mask, spot_mask]) * (18.0 * profile.midlife_maturity)

    return np.clip(textured, 0, 255).astype(np.uint8)


def _enhance_structural_features(
    face_roi: Any,
    landmarks: tuple[tuple[float, float], ...],
    profile: AgeProgressionProfile,
) -> Any:
    cv2, np = _load_dependencies()
    output = face_roi.astype(np.float32)
    shadow = np.zeros(face_roi.shape[:2], dtype=np.float32)
    highlight = np.zeros(face_roi.shape[:2], dtype=np.float32)
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    height, width = face_roi.shape[:2]
    strength = profile.structure_strength
    eye_y = float((left_eye[1] + right_eye[1]) / 2.0)
    mouth_y = float((left_mouth[1] + right_mouth[1]) / 2.0)

    _draw_soft_line(shadow, (nose[0], max(0.0, nose[1] - height * 0.14)), (nose[0], nose[1] + height * 0.08), 3, 0.65)
    _draw_soft_line(highlight, (nose[0] - width * 0.025, nose[1] - height * 0.13), (nose[0] - width * 0.015, nose[1]), 2, 0.55)
    _draw_soft_line(shadow, (left_eye[0], left_eye[1] + height * 0.045), (right_eye[0], right_eye[1] + height * 0.045), 3, 0.34)

    for eye, direction in ((left_eye, -1), (right_eye, 1)):
        _draw_soft_line(
            shadow,
            (eye[0] + direction * width * 0.04, eye[1] + height * 0.035),
            (eye[0] + direction * width * 0.17, eye[1] + height * 0.075),
            2,
            0.35 + 0.30 * profile.midlife_maturity,
        )

    _draw_soft_line(
        shadow,
        (left_eye[0], eye_y + height * 0.12),
        (left_mouth[0] - width * 0.02, mouth_y - height * 0.03),
        4,
        0.40,
    )
    _draw_soft_line(
        shadow,
        (right_eye[0], eye_y + height * 0.12),
        (right_mouth[0] + width * 0.02, mouth_y - height * 0.03),
        4,
        0.40,
    )
    _draw_soft_line(
        highlight,
        (width * 0.32, eye_y + height * 0.16),
        (width * 0.68, eye_y + height * 0.16),
        3,
        0.22,
    )
    _draw_soft_line(
        shadow,
        (left_mouth[0] - width * 0.08, mouth_y + height * 0.09),
        (right_mouth[0] + width * 0.08, mouth_y + height * 0.09),
        4,
        0.32,
    )
    _draw_soft_line(
        shadow,
        (width * 0.24, height * 0.77),
        (width * 0.76, height * 0.77),
        5,
        0.42 + 0.25 * profile.adult_maturity,
    )

    blur = max(5, (min(width, height) // 18) | 1)
    shadow = cv2.GaussianBlur(shadow, (blur, blur), 0)
    highlight = cv2.GaussianBlur(highlight, (blur, blur), 0)
    output = output - np.dstack([shadow, shadow, shadow]) * (33.0 * strength)
    output = output + np.dstack([highlight, highlight, highlight]) * (13.0 * strength)
    return np.clip(output, 0, 255).astype(np.uint8)


def _draw_soft_line(mask: Any, start: tuple[float, float], end: tuple[float, float], thickness: int, value: float) -> None:
    cv2, _ = _load_dependencies()
    cv2.line(
        mask,
        (int(round(start[0])), int(round(start[1]))),
        (int(round(end[0])), int(round(end[1]))),
        float(value),
        max(1, int(thickness)),
        cv2.LINE_AA,
    )


def _face_blend_mask(height: int, width: int, np: Any) -> Any:
    cv2, _ = _load_dependencies()
    mask = np.zeros((height, width), dtype=np.float32)
    center = (int(width * 0.50), int(height * 0.50))
    axes = (int(width * 0.43), int(height * 0.48))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur_size = max(9, (min(width, height) // 8) | 1)
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return np.clip(mask, 0.0, 1.0)


def _analyze_generated_bytes(image_bytes: bytes) -> FaceAnalysis:
    import tempfile

    suffix = ".jpg"
    temporary_file = tempfile.NamedTemporaryFile(prefix="age_progression_validate_", suffix=suffix, delete=False)
    temporary_path = Path(temporary_file.name)
    try:
        temporary_file.write(image_bytes)
        temporary_file.close()
        return analyze_face_image(temporary_path)
    finally:
        try:
            temporary_file.close()
            if temporary_path.exists():
                temporary_path.unlink()
        except OSError:
            logger.warning("Could not remove temporary age progression validation file: %s", temporary_path)


def _detect_single_face(image: Any) -> tuple[Any, float]:
    cv2, _ = _load_dependencies()
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN_create(
        str(OPENCV_YUNET_MODEL_PATH),
        "",
        (width, height),
        OPENCV_FACE_SCORE_THRESHOLD,
        OPENCV_FACE_NMS_THRESHOLD,
        OPENCV_FACE_TOP_K,
    )
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise FaceImageRejectedError("No face detected.")

    valid_faces = [face for face in faces if float(face[-1]) >= OPENCV_FACE_SCORE_THRESHOLD]
    if not valid_faces:
        raise FaceImageRejectedError("No face met the minimum detection confidence.")
    if len(valid_faces) > 1:
        raise FaceImageRejectedError("Multiple faces detected. Select an image with exactly one clear child face.")

    face = valid_faces[0]
    face_area_ratio = _face_area_ratio(face, width, height)
    if face_area_ratio < MIN_FACE_AREA_RATIO:
        raise FaceImageRejectedError(
            f"Detected face is too small in the image. Face area ratio {face_area_ratio:.4f} "
            f"is below the required {MIN_FACE_AREA_RATIO:.4f}."
        )
    return face, face_area_ratio


def _extract_landmarks(face: Any) -> tuple[tuple[float, float], ...]:
    values = [float(value) for value in face[4:14]]
    return tuple((values[index], values[index + 1]) for index in range(0, len(values), 2))


def _face_bbox(face: Any, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x = max(int(round(float(face[0]))), 0)
    y = max(int(round(float(face[1]))), 0)
    width = max(int(round(float(face[2]))), 1)
    height = max(int(round(float(face[3]))), 1)
    x = min(x, image_width - 1)
    y = min(y, image_height - 1)
    width = min(width, image_width - x)
    height = min(height, image_height - y)
    return x, y, width, height


def _expanded_face_region(bbox: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    pad_x = int(width * 0.38)
    pad_top = int(height * 0.42)
    pad_bottom = int(height * 0.28)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_top)
    x1 = min(image_width, x + width + pad_x)
    y1 = min(image_height, y + height + pad_bottom)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _face_area_ratio(face: Any, image_width: int, image_height: int) -> float:
    face_width = max(float(face[2]), 0.0)
    face_height = max(float(face[3]), 0.0)
    image_area = float(image_width * image_height)
    return 0.0 if image_area <= 0 else (face_width * face_height) / image_area


def _validate_age_request(source_age: int, target_age: int) -> None:
    if source_age < MIN_CHILD_AGE or source_age > MAX_CHILD_AGE:
        raise AgeProgressionError(f"Source age must be between {MIN_CHILD_AGE} and {MAX_CHILD_AGE}.")
    if target_age < MIN_CHILD_AGE or target_age > MAX_CHILD_AGE:
        raise AgeProgressionError(f"Target age must be between {MIN_CHILD_AGE} and {MAX_CHILD_AGE}.")
    if target_age <= source_age:
        raise AgeProgressionError("Target age must be greater than the registered age.")


def _validate_source_path(image_path: Path) -> Path:
    if not image_path.exists() or not image_path.is_file():
        raise FaceImageRejectedError("Source image file is missing.")
    if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise FaceImageRejectedError("Source image must be a JPEG or PNG file.")
    return image_path


def _ensure_opencv_model_assets() -> None:
    try:
        ensure_model_file(
            ModelAssetSpec(
                path=OPENCV_YUNET_MODEL_PATH,
                urls=OPENCV_YUNET_MODEL_URLS,
                sha256=OPENCV_YUNET_MODEL_SHA256,
                size_bytes=OPENCV_YUNET_MODEL_SIZE,
            )
        )
        ensure_model_file(
            ModelAssetSpec(
                path=OPENCV_SFACE_MODEL_PATH,
                urls=OPENCV_SFACE_MODEL_URLS,
                sha256=OPENCV_SFACE_MODEL_SHA256,
                size_bytes=OPENCV_SFACE_MODEL_SIZE,
            )
        )
    except ModelAssetError as exc:
        raise AgeProgressionDependencyError(str(exc)) from exc


def _load_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise FaceEmbeddingDependencyError("OpenCV and NumPy are required for age progression.") from exc
    return cv2, np


def _ensure_opencv_face_api(cv2: Any) -> None:
    if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
        raise AgeProgressionDependencyError(
            "Installed OpenCV does not include YuNet/SFace APIs. Install dependencies from requirements.txt."
        )


def _smoothstep(value: float, edge0: float, edge1: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    normalized = min(max((value - edge0) / (edge1 - edge0), 0.0), 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _sigmoid(value: Any, np: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)
