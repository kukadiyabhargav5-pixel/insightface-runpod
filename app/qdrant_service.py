"""
Qdrant vector database integration for wedding & event face indexing and similarity search.
Uses Cosine distance with dynamic dimension verification and payload filtering.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

logger = logging.getLogger("insightface-worker.qdrant")

# Stable namespace for deterministic UUIDv5 point ID generation
NAMESPACE_MARA_PHOTO = uuid.UUID("7a2e7c65-8b1e-49c9-94b2-e1d5a7d65b21")


class QdrantService:
    """
    Manages collection lifecycle, idempotent upserts, and filtered vector search in Qdrant.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        self.url = url or os.getenv("QDRANT_URL", "").strip()
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", "").strip() or None
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "face_embeddings").strip()

        if not self.url:
            logger.warning("QDRANT_URL environment variable is not configured. Vector operations will fail if invoked.")
            self.client = None
        else:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30.0
            )
            logger.info(f"Connected to Qdrant at {self._safe_url_log(self.url)} (Collection: '{self.collection_name}')")

    @staticmethod
    def _safe_url_log(url: str) -> str:
        """
        Masks passwords if embedded in connection string.
        """
        try:
            from urllib.parse import urlsplit, urlunsplit
            parsed = urlsplit(url)
            netloc = parsed.hostname or ""
            if parsed.port:
                netloc += f":{parsed.port}"
            return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        except Exception:
            return "<masked-url>"

    def is_configured(self) -> bool:
        return self.client is not None

    def ensure_collection(self, dimension: int) -> bool:
        """
        Creates the Qdrant collection if it does not already exist.
        Matches the vector size to the exact InsightFace model embedding dimension.
        """
        if not self.client:
            raise RuntimeError("QdrantClient is not configured. Set QDRANT_URL environment variable.")

        try:
            exists = self.client.collection_exists(collection_name=self.collection_name)
            if not exists:
                logger.info(
                    f"Collection '{self.collection_name}' does not exist. "
                    f"Creating with dimension={dimension}, distance=Cosine..."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(
                        size=dimension,
                        distance=rest.Distance.COSINE
                    )
                )

                # Create payload indexes for high-speed keyword filtering
                for field in ["photo_id", "album_id"]:
                    try:
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field,
                            field_schema=rest.PayloadSchemaType.KEYWORD
                        )
                        logger.info(f"Created keyword index on '{field}' in collection '{self.collection_name}'.")
                    except Exception as idx_err:
                        logger.debug(f"Payload index creation note: {idx_err}")

                logger.info(f"Collection '{self.collection_name}' created successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection '{self.collection_name}': {e}")
            raise

    @staticmethod
    def generate_point_id(photo_id: str, face_index: int) -> str:
        """
        Generates a deterministic UUIDv5 based on photo_id and face_index.
        Ensures idempotent upserts without creating duplicate vectors if re-indexed.
        """
        unique_key = f"{photo_id}_{face_index}"
        return str(uuid.uuid5(NAMESPACE_MARA_PHOTO, unique_key))

    def upsert_faces(
        self,
        photo_id: str,
        faces: List[Dict[str, Any]],
        album_id: Optional[str] = None
    ) -> int:
        """
        Upserts all detected face embeddings for a photo into Qdrant.
        Returns the count of points indexed.
        """
        if not self.client:
            raise RuntimeError("QdrantClient is not configured.")

        if not faces:
            return 0

        # Verify dimension on first face
        dimension = faces[0]["embedding_dimension"]
        self.ensure_collection(dimension=dimension)

        points: List[rest.PointStruct] = []
        for face in faces:
            point_id = self.generate_point_id(photo_id, face["face_index"])
            payload = {
                "photo_id": photo_id,
                "album_id": album_id,
                "face_index": face["face_index"],
                "bbox": face["bbox"],
                "det_score": face["det_score"],
                "source": "insightface"
            }
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=face["embedding"],
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        logger.info(f"Indexed {len(points)} face vector(s) for photo_id='{photo_id}' into '{self.collection_name}'.")
        return len(points)

    def search_similar_faces(
        self,
        query_embedding: List[float],
        limit: int = 25,
        score_threshold: float = 0.45,
        album_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search in Qdrant against indexed face embeddings.
        Supports filtering by album_id.
        Deduplicates results by photo_id (keeping the highest similarity score).
        """
        if not self.client:
            raise RuntimeError("QdrantClient is not configured.")

        # Ensure collection exists before querying
        self.ensure_collection(dimension=len(query_embedding))

        # Optional payload filter
        query_filter: Optional[rest.Filter] = None
        if album_id:
            query_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="album_id",
                        match=rest.MatchValue(value=album_id)
                    )
                ]
            )

        # Execute vector search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit * 2,  # Fetch extra to account for multiple faces in the same photo before deduplication
            score_threshold=score_threshold
        )

        # Deduplicate matches by photo_id (keep highest similarity score per photo)
        photo_map: Dict[str, Dict[str, Any]] = {}
        for hit in search_results:
            payload = hit.payload or {}
            photo_id = payload.get("photo_id")
            if not photo_id:
                continue

            current_score = float(hit.score)
            if photo_id not in photo_map or current_score > photo_map[photo_id]["score"]:
                photo_map[photo_id] = {
                    "photo_id": photo_id,
                    "album_id": payload.get("album_id"),
                    "face_index": payload.get("face_index"),
                    "score": round(current_score, 4),
                    "bbox": payload.get("bbox")
                }

        # Sort deduplicated results descending by score and slice by limit
        sorted_matches = sorted(photo_map.values(), key=lambda x: x["score"], reverse=True)[:limit]
        logger.info(
            f"Vector search returned {len(sorted_matches)} unique photo match(es) "
            f"(threshold={score_threshold}, album_filter={album_id})."
        )
        return sorted_matches

    def delete_by_photo_id(self, photo_id: str) -> int:
        """
        Deletes all face vectors belonging to a given photo_id.
        """
        if not self.client:
            raise RuntimeError("QdrantClient is not configured.")

        filter_condition = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="photo_id",
                    match=rest.MatchValue(value=photo_id)
                )
            ]
        )

        res = self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.FilterSelector(filter=filter_condition),
            wait=True
        )
        logger.info(f"Deleted face vectors for photo_id='{photo_id}' from '{self.collection_name}'.")
        return 1

    def health_check(self) -> Dict[str, Any]:
        """
        Verifies connectivity and collection status in Qdrant cluster.
        """
        if not self.client:
            return {"configured": False, "status": "unconfigured"}

        try:
            collections_resp = self.client.get_collections()
            collection_names = [c.name for c in collections_resp.collections]
            collection_exists = self.collection_name in collection_names

            info = {}
            if collection_exists:
                c_info = self.client.get_collection(collection_name=self.collection_name)
                info = {
                    "points_count": getattr(c_info, "points_count", None),
                    "vectors_count": getattr(c_info, "vectors_count", None),
                    "status": getattr(c_info, "status", "ready")
                }

            return {
                "configured": True,
                "connected": True,
                "collection_name": self.collection_name,
                "collection_exists": collection_exists,
                "details": info
            }
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return {
                "configured": True,
                "connected": False,
                "error": str(e)
            }
