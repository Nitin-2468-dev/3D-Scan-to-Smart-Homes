"""
Pydantic models for the 3D Reconstruction + Multimodal QA API.

These models match the OpenAPI specification in specs/openapi.yaml.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# Enums
class SessionStatus(str, Enum):
    """Current session processing status."""

    CREATED = "created"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    RECONSTRUCTING = "reconstructing"
    STRUCTURING = "structuring"
    READY = "ready"
    FAILED = "failed"
    FALLBACK = "fallback"


class HealthStatus(str, Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ReconstructionMode(str, Enum):
    """Reconstruction quality mode."""

    QUICK = "quick"
    FULL = "full"


class JobStatus(str, Enum):
    """Job acceptance status."""

    ACCEPTED = "accepted"
    QUEUED = "queued"


class ArtifactType(str, Enum):
    """Type of generated artifact."""

    POINTCLOUD = "pointcloud"
    SCENE_STRUCTURE = "scene_structure"
    CAMERA_POSES = "camera_poses"
    INSTANCE_MASK = "instance_mask"
    DEPTH_MAP = "depth_map"
    MESH = "mesh"
    THUMBNAIL = "thumbnail"


class FallbackSceneType(str, Enum):
    """Type of fallback scene to load."""

    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    OFFICE = "office"


# Nested models
class Point3D(BaseModel):
    """3D point coordinates."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate")


class BoundingBox(BaseModel):
    """3D bounding box."""

    min_x: float = Field(..., description="Minimum X coordinate")
    min_y: float = Field(..., description="Minimum Y coordinate")
    min_z: float = Field(..., description="Minimum Z coordinate")
    max_x: float = Field(..., description="Maximum X coordinate")
    max_y: float = Field(..., description="Maximum Y coordinate")
    max_z: float = Field(..., description="Maximum Z coordinate")


class Instance(BaseModel):
    """Detected object instance."""

    id: str = Field(..., description="Unique instance identifier")
    class_name: str = Field(..., description="Detected object class")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Detection confidence score"
    )
    bounding_box: BoundingBox = Field(..., description="3D bounding box")
    center: Optional[Point3D] = Field(None, description="Center point of the object")
    mask_path: Optional[str] = Field(
        None, description="Path to instance segmentation mask"
    )
    embedding: Optional[list[float]] = Field(
        None, description="Instance embedding vector"
    )


class Artifact(BaseModel):
    """Generated artifact metadata."""

    name: str = Field(..., description="Artifact filename")
    type: ArtifactType = Field(..., description="Artifact type")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(..., description="Creation timestamp")
    download_url: Optional[str] = Field(None, description="Optional direct download URL")


# Request models
class CreateSessionRequest(BaseModel):
    """Request body for session creation (non-file fields)."""

    name: Optional[str] = Field(None, description="Optional session name")


class ReconstructionRequest(BaseModel):
    """Request body for triggering reconstruction."""

    mode: ReconstructionMode = Field(
        default=ReconstructionMode.QUICK,
        description="Reconstruction quality mode",
    )


class AskRequest(BaseModel):
    """Request body for multimodal QA."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to ask about the scene",
    )
    object_id: Optional[str] = Field(
        None, description="Optional object ID to focus the question on"
    )
    include_context: bool = Field(
        default=True,
        description="Whether to include scene context in the response",
    )


class FallbackRequest(BaseModel):
    """Request body for loading fallback assets."""

    scene_type: FallbackSceneType = Field(
        default=FallbackSceneType.LIVING_ROOM,
        description="Type of fallback scene to load",
    )


# Response models
class HealthResponse(BaseModel):
    """Health check response."""

    status: HealthStatus = Field(..., description="Current health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server timestamp")


class SessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str = Field(..., description="Unique session identifier")
    name: Optional[str] = Field(None, description="Optional session name")
    status: SessionStatus = Field(..., description="Current session status")
    image_count: int = Field(..., ge=0, description="Number of uploaded images")
    created_at: datetime = Field(..., description="Session creation timestamp")


class SessionDetailResponse(BaseModel):
    """Detailed session information."""

    session_id: str = Field(..., description="Unique session identifier")
    name: Optional[str] = Field(None, description="Optional session name")
    status: SessionStatus = Field(..., description="Current session status")
    image_count: int = Field(..., ge=0, description="Number of uploaded images")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    reconstruction_mode: Optional[ReconstructionMode] = Field(
        None, description="Reconstruction quality mode"
    )
    has_reconstruction: bool = Field(
        default=False, description="Whether reconstruction is complete"
    )
    has_structure: bool = Field(
        default=False, description="Whether scene structure is generated"
    )
    has_fallback: bool = Field(
        default=False, description="Whether fallback assets are loaded"
    )
    artifact_count: int = Field(default=0, description="Number of available artifacts")
    instance_count: int = Field(default=0, description="Number of detected instances")


class SessionStatusResponse(BaseModel):
    """Session processing status."""

    session_id: str = Field(..., description="Unique session identifier")
    status: SessionStatus = Field(..., description="Current session status")
    current_step: Optional[str] = Field(None, description="Current processing step")
    progress: Optional[int] = Field(
        None, ge=0, le=100, description="Progress percentage (0-100)"
    )
    message: Optional[str] = Field(None, description="Human-readable status message")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )
    error: Optional[str] = Field(
        None, description="Error message if status is failed"
    )


class JobAcceptedResponse(BaseModel):
    """Response for async job acceptance."""

    job_id: str = Field(..., description="Unique job identifier")
    session_id: str = Field(..., description="Associated session identifier")
    status: JobStatus = Field(..., description="Job acceptance status")
    message: Optional[str] = Field(None, description="Status message")


class AskResponse(BaseModel):
    """Response for multimodal QA."""

    session_id: str = Field(..., description="Session identifier")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Answer confidence score"
    )
    referenced_objects: Optional[list[str]] = Field(
        None, description="Object IDs referenced in the answer"
    )
    processing_time_ms: Optional[int] = Field(
        None, description="Processing time in milliseconds"
    )
    is_fallback: Optional[bool] = Field(
        None, description="Whether this answer came from fallback data"
    )


class ArtifactListResponse(BaseModel):
    """Response for listing artifacts."""

    session_id: str = Field(..., description="Session identifier")
    artifacts: list[Artifact] = Field(..., description="List of artifacts")


class InstanceListResponse(BaseModel):
    """Response for listing detected instances."""

    session_id: str = Field(..., description="Session identifier")
    instance_count: int = Field(..., description="Total number of instances")
    instances: list[Instance] = Field(..., description="List of instances")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for debugging")
