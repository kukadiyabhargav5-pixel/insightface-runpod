"""
Unit and integration test suite for InsightFace RunPod Serverless worker.
Validates schemas, error handling, face selection heuristics, and response structure.
Compatible with both pytest and Python standard library unittest.
"""

import os
import sys
import uuid
import tempfile
from unittest.mock import MagicMock, patch

# Ensure app package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas import HandlerInput, OperationType, FaceSelectionStrategy, DetectedFace
from app.image_loader import sanitize_url_for_logging, temp_image_context
from app.face_engine import FaceEngine
from app.qdrant_service import QdrantService, NAMESPACE_MARA_PHOTO


# ==============================================================================
# SCHEMA & VALIDATION TESTS
# ==============================================================================

def test_handler_input_defaults():
    payload = {"image_url": "https://r2.example.com/test.jpg"}
    data = HandlerInput(**payload)
    assert data.operation == OperationType.DETECT.value
    assert data.limit == 25
    assert data.score_threshold == 0.45
    assert data.face_selection == FaceSelectionStrategy.LARGEST.value


def test_handler_input_invalid_operation():
    payload = {"operation": "invalid_op", "image_url": "https://r2.example.com/test.jpg"}
    raised = False
    try:
        HandlerInput(**payload)
    except ValueError as exc:
        raised = True
        assert "Invalid operation" in str(exc)
    assert raised, "Expected ValueError was not raised for invalid operation"


def test_handler_input_face_selection_fallback():
    payload = {"operation": "search", "image_url": "https://r2.example.com/test.jpg", "face_selection": "non_existent"}
    data = HandlerInput(**payload)
    assert data.face_selection == FaceSelectionStrategy.LARGEST.value


# ==============================================================================
# URL SANITIZATION & SECURITY TESTS
# ==============================================================================

def test_sanitize_url_for_logging_strips_query_params():
    sensitive_url = (
        "https://pub-12345.r2.dev/events/wedding_01.jpg"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=SECRET%2F20261001"
        "&X-Amz-Signature=abcdef1234567890"
    )
    sanitized = sanitize_url_for_logging(sensitive_url)
    assert "SECRET" not in sanitized
    assert "Signature" not in sanitized
    assert "?" not in sanitized
    assert sanitized == "https://pub-12345.r2.dev/events/wedding_01.jpg"


def test_sanitize_url_empty_or_malformed():
    assert sanitize_url_for_logging("") == ""
    assert sanitize_url_for_logging(None) == ""


# ==============================================================================
# FACE ENGINE & SELECTION HEURISTICS
# ==============================================================================

def test_l2_normalization():
    import numpy as np
    raw_vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    normalized = FaceEngine.normalize_l2(raw_vec)
    norm = float(np.linalg.norm(normalized))
    assert abs(norm - 1.0) < 1e-5
    assert abs(float(normalized[0]) - 0.6) < 1e-5
    assert abs(float(normalized[1]) - 0.8) < 1e-5


def test_select_query_face_largest_vs_confidence():
    # Face 1: Large area, medium confidence
    face_large = {
        "face_index": 0,
        "bbox": [10.0, 10.0, 210.0, 210.0],  # 200x200 = 40,000 area
        "det_score": 0.88,
        "area": 40000.0,
        "embedding": [0.1] * 512
    }
    # Face 2: Small background face, high confidence
    face_small = {
        "face_index": 1,
        "bbox": [300.0, 300.0, 350.0, 350.0],  # 50x50 = 2,500 area
        "det_score": 0.99,
        "area": 2500.0,
        "embedding": [0.2] * 512
    }
    faces = [face_large, face_small]

    # Test largest (default heuristic)
    selected_largest = FaceEngine.select_query_face(faces, strategy="largest")
    assert selected_largest["face_index"] == 0

    # Test highest confidence heuristic
    selected_conf = FaceEngine.select_query_face(faces, strategy="highest_confidence")
    assert selected_conf["face_index"] == 1


def test_select_query_face_empty_raises_error():
    raised = False
    try:
        FaceEngine.select_query_face([], strategy="largest")
    except ValueError as exc:
        raised = True
        assert "No faces detected" in str(exc)
    assert raised, "Expected ValueError was not raised for empty faces list"


# ==============================================================================
# QDRANT DETERMINISTIC ID & IDEMPOTENCY
# ==============================================================================

def test_generate_point_id_idempotency():
    photo_id = "photo_wedding_481"
    face_index = 0

    id_1 = QdrantService.generate_point_id(photo_id, face_index)
    id_2 = QdrantService.generate_point_id(photo_id, face_index)

    # Must be identical UUIDv5
    assert id_1 == id_2
    assert uuid.UUID(id_1).version == 5

    # Different face_index must yield different point ID
    id_face_1 = QdrantService.generate_point_id(photo_id, 1)
    assert id_1 != id_face_1


# ==============================================================================
# TEMPORARY IMAGE CONTEXT & CLEANUP GUARANTEE
# ==============================================================================

def test_temp_image_cleanup_on_exception():
    """
    Verifies that temporary image files are deleted even if an unexpected error occurs during processing.
    """
    created_temp_path = None

    with patch("app.image_loader.download_temporary_image") as mock_download:
        # Create a real dummy file to simulate the temp file on disk
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(b"dummy image bytes")
            created_temp_path = tf.name

        mock_download.return_value = created_temp_path

        with patch("cv2.imread") as mock_imread:
            import numpy as np
            mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            try:
                with temp_image_context("https://example.com/test.jpg") as img:
                    assert img is not None
                    # Simulate processing exception
                    raise RuntimeError("Simulated crash during inference")
            except RuntimeError:
                pass

    # Verify temp file was deleted in finally block
    assert not os.path.exists(created_temp_path)


# ==============================================================================
# RUNPOD HANDLER EXECUTION TESTS
# ==============================================================================

@patch("handler.face_engine")
@patch("handler.qdrant_service")
def test_handler_missing_image_url(mock_qdrant, mock_engine):
    from handler import handler

    event = {"input": {"operation": "detect"}}
    response = handler(event)

    assert response["success"] is False
    assert "image_url" in response["error"]
    assert "processing_time_ms" in response


@patch("handler.face_engine")
@patch("handler.qdrant_service")
def test_handler_index_missing_photo_id(mock_qdrant, mock_engine):
    from handler import handler

    event = {
        "input": {
            "operation": "index",
            "image_url": "https://example.com/test.jpg"
        }
    }
    response = handler(event)

    assert response["success"] is False
    assert "photo_id" in response["error"]


@patch("handler.face_engine")
@patch("handler.qdrant_service")
def test_handler_search_no_faces(mock_qdrant, mock_engine):
    from handler import handler

    mock_qdrant.is_configured.return_value = True
    mock_engine.extract_faces.return_value = []

    with patch("handler.temp_image_context") as mock_ctx:
        import numpy as np
        mock_ctx.return_value.__enter__.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_ctx.return_value.__exit__.return_value = None

        event = {
            "input": {
                "operation": "search",
                "image_url": "https://example.com/selfie.jpg",
                "album_id": "album_123"
            }
        }
        response = handler(event)

        assert response["success"] is True
        assert response["operation"] == "search"
        assert response["face_count"] == 0
        assert response["matches"] == []


if __name__ == "__main__":
    print("Running insightface-runpod test suite...")
    test_handler_input_defaults()
    test_handler_input_invalid_operation()
    test_handler_input_face_selection_fallback()
    test_sanitize_url_for_logging_strips_query_params()
    test_sanitize_url_empty_or_malformed()
    test_l2_normalization()
    test_select_query_face_largest_vs_confidence()
    test_select_query_face_empty_raises_error()
    test_generate_point_id_idempotency()
    test_temp_image_cleanup_on_exception()
    test_handler_missing_image_url()
    test_handler_index_missing_photo_id()
    test_handler_search_no_faces()
    print("All 13 unit tests passed successfully!")
