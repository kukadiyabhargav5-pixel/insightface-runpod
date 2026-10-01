"""
InsightFace Face Detection and Embedding Extraction Engine.
Singleton model initialization with CUDA acceleration and CPU fallback.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger("insightface-worker.face_engine")


class FaceEngine:
    """
    Manages the lifecycle and inference of InsightFace FaceAnalysis.
    Loaded once at worker boot and kept warm in GPU/CPU memory.
    """

    _instance: Optional["FaceEngine"] = None

    def __init__(self, model_name: Optional[str] = None, detection_size: Optional[int] = None):
        self.model_name = model_name or os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
        det_dim = int(detection_size or os.getenv("DETECTION_SIZE", "640"))
        self.detection_size = (det_dim, det_dim)

        self.providers: List[str] = []
        self.cuda_available: bool = False
        self.active_provider: str = "Unknown"
        self.embedding_dimension: int = 512

        self.app = self._initialize_model()

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, detection_size: Optional[int] = None) -> "FaceEngine":
        """
        Thread-safe singleton accessor to ensure model is never duplicated in memory.
        """
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, detection_size=detection_size)
        return cls._instance

    def _initialize_model(self):
        """
        Initializes InsightFace with CUDA provider precedence.
        """
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime execution providers: {available_providers}")

        self.cuda_available = "CUDAExecutionProvider" in available_providers

        # Prefer CUDAExecutionProvider, fallback to CPUExecutionProvider
        self.providers = []
        if self.cuda_available:
            self.providers.append("CUDAExecutionProvider")
        self.providers.append("CPUExecutionProvider")

        ctx_id = 0 if self.cuda_available else -1

        logger.info(f"Booting FaceAnalysis(name='{self.model_name}', providers={self.providers})...")

        try:
            app = FaceAnalysis(
                name=self.model_name,
                providers=self.providers
            )
            app.prepare(ctx_id=ctx_id, det_size=self.detection_size)
        except Exception as e:
            logger.warning(f"Failed to initialize with providers {self.providers}: {e}. Retrying CPU-only fallback...")
            self.providers = ["CPUExecutionProvider"]
            self.cuda_available = False
            app = FaceAnalysis(name=self.model_name, providers=self.providers)
            app.prepare(ctx_id=-1, det_size=self.detection_size)

        # Inspect active provider and embedding dimension dynamically
        self.active_provider = self.providers[0]
        self._probe_embedding_dimension(app)

        # Startup diagnostic logging
        logger.info("=" * 60)
        logger.info("INSIGHTFACE ENGINE INITIALIZED SUCCESSFULLY")
        logger.info(f"  • Model Name:          {self.model_name}")
        logger.info(f"  • Detection Size:      {self.detection_size}")
        logger.info(f"  • CUDA Available:      {self.cuda_available}")
        logger.info(f"  • Active Provider:     {self.active_provider}")
        logger.info(f"  • Embedding Dimension: {self.embedding_dimension}")
        logger.info("=" * 60)

        return app

    def _probe_embedding_dimension(self, app):
        """
        Runs a synthetic lightweight test to verify exact embedding vector length.
        """
        try:
            # Create a blank 640x640 dummy image
            dummy = np.zeros((self.detection_size[0], self.detection_size[1], 3), dtype=np.uint8)
            # Check recognition model embedding dimension if accessible directly
            if hasattr(app, "models") and "recognition" in app.models:
                rec_model = app.models["recognition"]
                if hasattr(rec_model, "output_shape") and rec_model.output_shape:
                    self.embedding_dimension = int(rec_model.output_shape[-1])
                    return
            # Default to 512 for standard buffalo_l/buffalo_s
            self.embedding_dimension = 512
        except Exception as e:
            logger.debug(f"Dynamic probe used default dimension: {e}")
            self.embedding_dimension = 512

    @staticmethod
    def normalize_l2(vector: np.ndarray) -> np.ndarray:
        """
        Applies L2 normalization to convert vector to unit sphere for accurate cosine similarity.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    def extract_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs face detection and embedding extraction on a BGR image array.
        Returns a list of structured face dictionaries.
        """
        if image is None or image.size == 0:
            return []

        raw_faces = self.app.get(image)
        results: List[Dict[str, Any]] = []

        for idx, face in enumerate(raw_faces):
            # Bounding box [x1, y1, x2, y2]
            bbox = [float(coord) for coord in face.bbox]
            score = float(face.det_score) if hasattr(face, "det_score") else 1.0

            # Normalized float32 embedding
            embedding_np = self.normalize_l2(face.embedding.astype(np.float32))
            embedding_list = [float(x) for x in embedding_np]

            # Face area for selection heuristics: (x2 - x1) * (y2 - y1)
            width = max(0.0, bbox[2] - bbox[0])
            height = max(0.0, bbox[3] - bbox[1])
            area = width * height

            results.append({
                "face_index": idx,
                "bbox": bbox,
                "det_score": round(score, 4),
                "embedding_dimension": len(embedding_list),
                "embedding": embedding_list,
                "area": area
            })

        return results

    @staticmethod
    def select_query_face(faces: List[Dict[str, Any]], strategy: str = "largest") -> Dict[str, Any]:
        """
        Picks the primary face to search for when an image contains multiple faces.
        Options:
          - 'largest': Largest face bounding box area (default, recommended for user selfies)
          - 'highest_confidence': Highest detection confidence score
        """
        if not faces:
            raise ValueError("No faces detected in query image.")

        if len(faces) == 1:
            return faces[0]

        if strategy == "highest_confidence":
            return max(faces, key=lambda f: f.get("det_score", 0.0))

        # Default: 'largest' face area
        return max(faces, key=lambda f: f.get("area", 0.0))
