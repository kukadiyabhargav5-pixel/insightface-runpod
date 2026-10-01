"""
Data schemas and request/response models for InsightFace RunPod worker.
"""

from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, HttpUrl, field_validator


class OperationType(str, Enum):
    DETECT = "detect"
    INDEX = "index"
    SEARCH = "search"
    HEALTH = "health"
    DELETE = "delete"


class FaceSelectionStrategy(str, Enum):
    LARGEST = "largest"
    HIGHEST_CONFIDENCE = "highest_confidence"


class HandlerInput(BaseModel):
    """
    Standardized payload received in the RunPod 'input' dictionary.
    """
    operation: str = Field(default=OperationType.DETECT.value, description="Operation to perform: detect, index, search, health, delete")
    image_url: Optional[str] = Field(default=None, description="HTTP/HTTPS URL pointing to image (e.g. Cloudflare R2)")
    photo_id: Optional[str] = Field(default=None, description="Unique photo identifier for indexing or deletion")
    album_id: Optional[str] = Field(default=None, description="Optional album or event identifier for filtering")
    limit: int = Field(default=25, ge=1, le=100, description="Max matches returned in search mode")
    score_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="Minimum cosine similarity score threshold")
    face_selection: str = Field(default=FaceSelectionStrategy.LARGEST.value, description="Strategy for query face in search: 'largest' or 'highest_confidence'")

    @field_validator("operation")
    @classmethod
    def normalize_operation(cls, v: str) -> str:
        val = (v or "").strip().lower()
        valid = [op.value for op in OperationType]
        if val not in valid:
            raise ValueError(f"Invalid operation '{v}'. Supported operations: {valid}")
        return val

    @field_validator("face_selection")
    @classmethod
    def normalize_selection(cls, v: str) -> str:
        val = (v or "").strip().lower()
        valid = [s.value for s in FaceSelectionStrategy]
        if val not in valid:
            return FaceSelectionStrategy.LARGEST.value
        return val


class DetectedFace(BaseModel):
    """
    Metadata and embedding vector for an individual detected face.
    """
    face_index: int
    bbox: List[float]  # [x1, y1, x2, y2]
    det_score: float
    embedding_dimension: int
    embedding: List[float]


class SearchMatch(BaseModel):
    """
    Vector search match result.
    """
    photo_id: str
    album_id: Optional[str] = None
    face_index: Optional[int] = None
    score: float
    bbox: Optional[List[float]] = None


class WorkerResponse(BaseModel):
    """
    Standard response format for the RunPod handler.
    """
    success: bool
    operation: str
    face_count: Optional[int] = None
    faces: Optional[List[DetectedFace]] = None
    matches: Optional[List[SearchMatch]] = None
    photo_id: Optional[str] = None
    album_id: Optional[str] = None
    faces_indexed: Optional[int] = None
    deleted_count: Optional[int] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
