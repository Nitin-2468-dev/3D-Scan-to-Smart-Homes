"""
Celery task stubs for async processing.

These are placeholder implementations that simulate the actual
reconstruction, segmentation, and LMM inference pipelines.
"""

import time
import uuid
from typing import Literal

from celery import Celery

from .config import get_settings

settings = get_settings()

# Initialize Celery app
celery_app = Celery(
    "tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Task time limits (in seconds)
TASK_TIME_LIMIT = 3600  # 1 hour max
TASK_SOFT_TIME_LIMIT = 3000  # 50 minutes soft limit

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=TASK_TIME_LIMIT,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
)


@celery_app.task(bind=True, name="tasks.run_reconstruction")
def run_reconstruction(
    self,
    session_id: str,
    mode: Literal["quick", "full"] = "quick",
) -> dict:
    """
    Simulates the 3D reconstruction pipeline.

    In production, this would:
    1. Load uploaded images
    2. Run IGGT or alternative reconstruction
    3. Generate point cloud and camera poses
    4. Save artifacts to storage

    Args:
        session_id: The session identifier
        mode: Reconstruction mode (quick or full)

    Returns:
        dict with reconstruction results
    """
    # Simulate processing time based on mode
    iterations = (
        settings.reconstruction_quick_iterations
        if mode == "quick"
        else settings.reconstruction_full_iterations
    )

    total_steps = 5
    steps = [
        "Loading images",
        "Extracting features",
        "Computing camera poses",
        "Generating point cloud",
        "Saving artifacts",
    ]

    for i, step in enumerate(steps):
        # Update task state for progress tracking
        self.update_state(
            state="PROGRESS",
            meta={
                "current_step": step,
                "progress": int((i / total_steps) * 100),
                "message": f"Step {i + 1}/{total_steps}: {step}",
            },
        )
        # Simulate processing time
        time.sleep(0.5 if mode == "quick" else 1.0)

    # Return mock results
    return {
        "session_id": session_id,
        "status": "completed",
        "mode": mode,
        "artifacts": [
            {"name": "pointcloud.ply", "type": "pointcloud", "size_bytes": 15234567},
            {"name": "poses.json", "type": "camera_poses", "size_bytes": 12345},
        ],
        "point_count": 150000 if mode == "quick" else 500000,
        "processing_time_seconds": 2.5 if mode == "quick" else 5.0,
    }


@celery_app.task(bind=True, name="tasks.run_scene_structuring")
def run_scene_structuring(self, session_id: str) -> dict:
    """
    Simulates scene structure generation.

    In production, this would:
    1. Load reconstruction data
    2. Run instance segmentation (SAM)
    3. Classify detected objects
    4. Extract layout elements
    5. Generate scene.json

    Args:
        session_id: The session identifier

    Returns:
        dict with scene structure results
    """
    total_steps = 4
    steps = [
        "Loading reconstruction",
        "Running instance segmentation",
        "Classifying objects",
        "Generating scene structure",
    ]

    for i, step in enumerate(steps):
        self.update_state(
            state="PROGRESS",
            meta={
                "current_step": step,
                "progress": int((i / total_steps) * 100),
                "message": f"Step {i + 1}/{total_steps}: {step}",
            },
        )
        time.sleep(0.5)

    # Return mock scene structure
    return {
        "session_id": session_id,
        "status": "completed",
        "instance_count": 8,
        "layout_elements": ["floor", "ceiling", "wall_north", "wall_south"],
        "artifacts": [
            {"name": "scene.json", "type": "scene_structure", "size_bytes": 45678},
        ],
        "processing_time_seconds": 2.0,
    }


@celery_app.task(bind=True, name="tasks.run_lmm_inference")
def run_lmm_inference(
    self,
    session_id: str,
    prompt: str,
    object_id: str | None = None,
) -> dict:
    """
    Simulates LMM inference for multimodal QA.

    In production, this would:
    1. Load scene data and relevant images
    2. Optionally crop to specific object
    3. Run LMM inference (Qwen2-VL or similar)
    4. Return structured answer

    Args:
        session_id: The session identifier
        prompt: The user's question
        object_id: Optional object ID to focus on

    Returns:
        dict with QA results
    """
    self.update_state(
        state="PROGRESS",
        meta={
            "current_step": "Running LMM inference",
            "progress": 50,
            "message": "Processing your question...",
        },
    )

    # Simulate inference time
    time.sleep(1.0)

    # Generate mock answer based on prompt
    mock_answers = {
        "furniture": "I can see a gray sofa, a wooden coffee table, two armchairs, a floor lamp, and a bookshelf against the wall.",
        "color": "The room has warm neutral tones with beige walls, a gray sofa, and wooden furniture in natural oak finish.",
        "count": "There are 8 objects detected in this scene: 1 sofa, 1 coffee table, 2 armchairs, 1 floor lamp, 1 bookshelf, 1 rug, and 1 TV stand.",
        "default": "This appears to be a modern living room with comfortable seating, good lighting, and functional furniture arrangement.",
    }

    # Simple keyword matching for mock response
    prompt_lower = prompt.lower()
    if "furniture" in prompt_lower or "object" in prompt_lower:
        answer = mock_answers["furniture"]
    elif "color" in prompt_lower or "style" in prompt_lower:
        answer = mock_answers["color"]
    elif "how many" in prompt_lower or "count" in prompt_lower:
        answer = mock_answers["count"]
    else:
        answer = mock_answers["default"]

    return {
        "session_id": session_id,
        "question": prompt,
        "answer": answer,
        "confidence": 0.87,
        "referenced_objects": ["obj_001", "obj_002", "obj_003"] if object_id else [],
        "processing_time_ms": 1250,
        "is_fallback": False,
    }


@celery_app.task(name="tasks.cleanup_session")
def cleanup_session(session_id: str) -> dict:
    """
    Cleanup task to remove session data after TTL expires.

    Args:
        session_id: The session identifier to cleanup

    Returns:
        dict with cleanup results
    """
    # In production, this would delete files and database records
    return {
        "session_id": session_id,
        "status": "cleaned",
        "message": f"Session {session_id} has been cleaned up",
    }
