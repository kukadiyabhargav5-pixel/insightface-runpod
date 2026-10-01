"""
RunPod Serverless Handler for InsightFace Face Recognition & Embedding Service.
Handles 'detect', 'index', 'search', 'delete', and 'health' operations.
"""

import os
import time
import logging
import traceback
from typing import Dict, Any

import runpod

from app.schemas import HandlerInput, OperationType
from app.face_engine import FaceEngine
from app.image_loader import temp_image_context, sanitize_url_for_logging
from app.qdrant_service import QdrantService

# Configure structured logging
LOG_LEVEL = logging.INFO
if os.getenv("RUNPOD_DEBUG", "false").lower() == "true":
    LOG_LEVEL = logging.DEBUG

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("insightface-worker")

# ==============================================================================
# WORKER COLD-START INITIALIZATION
# ==============================================================================
# Model is initialized once when the worker container boots and kept warm in RAM/VRAM.
logger.info(">>> Initializing RunPod Serverless InsightFace Worker...")

try:
    face_engine = FaceEngine.get_instance()
    qdrant_service = QdrantService()
    logger.info(">>> Worker dependencies initialized successfully. Ready for invocations.")
except Exception as init_err:
    logger.critical(f"FATAL: Worker cold-start initialization failed: {init_err}")
    logger.critical(traceback.format_exc())
    raise


# ==============================================================================
# OPERATION DISPATCHERS
# ==============================================================================

def handle_detect_operation(params: HandlerInput, max_size_mb: int, timeout: int) -> Dict[str, Any]:
    """
    Downloads an image temporarily, extracts all faces & normalized embeddings,
    and returns them without storing anything in database or disk.
    """
    if not params.image_url:
        raise ValueError("Field 'image_url' is required for face detection.")

    logger.info(f"Detecting faces in: {sanitize_url_for_logging(params.image_url)}")

    with temp_image_context(params.image_url, max_size_mb=max_size_mb, timeout=timeout) as img:
        faces = face_engine.extract_faces(img)

    # Sanitize output (remove internal fields like 'area' before returning)
    clean_faces = []
    for f in faces:
        clean_faces.append({
            "face_index": f["face_index"],
            "bbox": f["bbox"],
            "det_score": f["det_score"],
            "embedding_dimension": f["embedding_dimension"],
            "embedding": f["embedding"]
        })

    logger.info(f"Detected {len(clean_faces)} face(s).")
    return {
        "success": True,
        "operation": OperationType.DETECT.value,
        "face_count": len(clean_faces),
        "faces": clean_faces
    }


def handle_index_operation(params: HandlerInput, max_size_mb: int, timeout: int) -> Dict[str, Any]:
    """
    Downloads photo from R2, generates face embeddings, and upserts all faces
    into Qdrant with deterministic UUIDs and metadata. Deletes temp image immediately.
    """
    if not params.image_url:
        raise ValueError("Field 'image_url' is required for indexing.")
    if not params.photo_id:
        raise ValueError("Field 'photo_id' is required for indexing.")

    if not qdrant_service.is_configured():
        raise RuntimeError("Qdrant is not configured. QDRANT_URL environment variable is required for index operations.")

    logger.info(
        f"Indexing photo_id='{params.photo_id}', album_id='{params.album_id or 'none'}' "
        f"from: {sanitize_url_for_logging(params.image_url)}"
    )

    with temp_image_context(params.image_url, max_size_mb=max_size_mb, timeout=timeout) as img:
        faces = face_engine.extract_faces(img)

    indexed_count = 0
    if faces:
        indexed_count = qdrant_service.upsert_faces(
            photo_id=params.photo_id,
            faces=faces,
            album_id=params.album_id
        )

    logger.info(f"Indexed {indexed_count} face vector(s) for photo_id='{params.photo_id}'.")
    return {
        "success": True,
        "operation": OperationType.INDEX.value,
        "photo_id": params.photo_id,
        "album_id": params.album_id,
        "faces_indexed": indexed_count,
        "face_count": len(faces)
    }


def handle_search_operation(params: HandlerInput, max_size_mb: int, timeout: int) -> Dict[str, Any]:
    """
    Takes a query selfie image, detects faces, selects the primary face (largest or highest confidence),
    and queries Qdrant for similar photos matching the embedding.
    """
    if not params.image_url:
        raise ValueError("Field 'image_url' is required for face search.")

    if not qdrant_service.is_configured():
        raise RuntimeError("Qdrant is not configured. QDRANT_URL environment variable is required for search operations.")

    logger.info(
        f"Searching faces with query image: {sanitize_url_for_logging(params.image_url)} "
        f"(album_filter='{params.album_id or 'all'}', strategy='{params.face_selection}')"
    )

    with temp_image_context(params.image_url, max_size_mb=max_size_mb, timeout=timeout) as img:
        detected_faces = face_engine.extract_faces(img)

    if not detected_faces:
        logger.info("Search finished: 0 faces detected in query image.")
        return {
            "success": True,
            "operation": OperationType.SEARCH.value,
            "face_count": 0,
            "matches": [],
            "message": "No face detected in query selfie image."
        }

    # Select single target face based on user preference ('largest' or 'highest_confidence')
    target_face = face_engine.select_query_face(detected_faces, strategy=params.face_selection)
    logger.info(
        f"Selected query face index {target_face['face_index']} (conf={target_face['det_score']}) "
        f"from {len(detected_faces)} detected face(s)."
    )

    matches = qdrant_service.search_similar_faces(
        query_embedding=target_face["embedding"],
        limit=params.limit,
        score_threshold=params.score_threshold,
        album_id=params.album_id
    )

    return {
        "success": True,
        "operation": OperationType.SEARCH.value,
        "matches": matches,
        "query_face": {
            "face_index": target_face["face_index"],
            "bbox": target_face["bbox"],
            "det_score": target_face["det_score"]
        },
        "total_matches": len(matches)
    }


def handle_delete_operation(params: HandlerInput) -> Dict[str, Any]:
    """
    Deletes all indexed face embeddings associated with a photo_id from Qdrant.
    """
    if not params.photo_id:
        raise ValueError("Field 'photo_id' is required for delete operation.")

    if not qdrant_service.is_configured():
        raise RuntimeError("Qdrant is not configured.")

    deleted = qdrant_service.delete_by_photo_id(photo_id=params.photo_id)
    return {
        "success": True,
        "operation": OperationType.DELETE.value,
        "photo_id": params.photo_id,
        "deleted_count": deleted
    }


def handle_health_operation() -> Dict[str, Any]:
    """
    Diagnostic health check reporting worker uptime, GPU execution status, and vector DB connectivity.
    """
    qdrant_status = qdrant_service.health_check()
    return {
        "success": True,
        "operation": OperationType.HEALTH.value,
        "worker": {
            "status": "healthy",
            "model_name": face_engine.model_name,
            "detection_size": face_engine.detection_size,
            "cuda_available": face_engine.cuda_available,
            "active_provider": face_engine.active_provider,
            "embedding_dimension": face_engine.embedding_dimension
        },
        "qdrant": qdrant_status
    }


# ==============================================================================
# MAIN RUNPOD HANDLER ENTRYPOINT
# ==============================================================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Official RunPod Serverless Handler function.
    Receives request payload, validates input, executes operation, and returns clean JSON.
    """
    start_time = time.perf_counter()
    raw_input = event.get("input", {})

    if not isinstance(raw_input, dict):
        return {
            "success": False,
            "error": "Malformed request. 'input' key must contain a JSON dictionary.",
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2)
        }

    # Configuration guardrails
    max_size_mb = int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))
    timeout = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

    try:
        # Validate inputs via Pydantic model
        params = HandlerInput(**raw_input)
        op = params.operation

        logger.info(f"Incoming RunPod request -> Operation: '{op}'")

        if op == OperationType.DETECT.value:
            res = handle_detect_operation(params, max_size_mb=max_size_mb, timeout=timeout)
        elif op == OperationType.INDEX.value:
            res = handle_index_operation(params, max_size_mb=max_size_mb, timeout=timeout)
        elif op == OperationType.SEARCH.value:
            res = handle_search_operation(params, max_size_mb=max_size_mb, timeout=timeout)
        elif op == OperationType.DELETE.value:
            res = handle_delete_operation(params)
        elif op == OperationType.HEALTH.value:
            res = handle_health_operation()
        else:
            raise ValueError(f"Unsupported operation '{op}'")

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        res["processing_time_ms"] = elapsed_ms
        logger.info(f"Operation '{op}' completed successfully in {elapsed_ms}ms.")
        return res

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        error_msg = str(exc)
        logger.error(f"Error handling request: {error_msg}")
        if os.getenv("RUNPOD_DEBUG", "false").lower() == "true":
            logger.error(traceback.format_exc())

        return {
            "success": False,
            "error": error_msg,
            "operation": raw_input.get("operation", "unknown"),
            "processing_time_ms": elapsed_ms
        }


# Start RunPod Serverless Worker loop
# NOTE: Must be at module level (NOT inside `if __name__ == "__main__":`)
# so that RunPod's repo scanner can detect the entry point.
runpod.serverless.start({"handler": handler})
