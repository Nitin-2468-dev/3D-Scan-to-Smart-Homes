"""
FastAPI application for 3D Reconstruction + Multimodal QA.

This module implements all endpoints from the OpenAPI specification.
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import ensure_directories, get_settings
from .models import (
    Artifact,
    ArtifactListResponse,
    ArtifactType,
    AskRequest,
    AskResponse,
    BoundingBox,
    ErrorResponse,
    FallbackRequest,
    FallbackSceneType,
    HealthResponse,
    HealthStatus,
    Instance,
    InstanceListResponse,
    JobAcceptedResponse,
    JobStatus,
    Point3D,
    ReconstructionMode,
    ReconstructionRequest,
    SessionDetailResponse,
    SessionResponse,
    SessionStatus,
    SessionStatusResponse,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    ensure_directories()
    yield
    # Shutdown
    pass


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for multi-view 3D reconstruction with instance segmentation and multimodal QA",
    lifespan=lifespan,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# In-memory session storage (for demo; production would use Redis/DB)
sessions: dict[str, dict] = {}


def generate_session_id() -> str:
    """Generate a unique session identifier."""
    return f"ses_{uuid.uuid4().hex[:12]}"


def generate_job_id() -> str:
    """Generate a unique job identifier."""
    return f"job_{uuid.uuid4().hex[:8]}"


def generate_request_id() -> str:
    """Generate a unique request identifier."""
    return f"req_{uuid.uuid4().hex[:8]}"


def get_fallback_data(scene_type: FallbackSceneType = FallbackSceneType.LIVING_ROOM) -> dict:
    """Load fallback scene data."""
    # Mock fallback data for living room
    return {
        "scene": {
            "name": f"Fallback {scene_type.value.replace('_', ' ').title()}",
            "bounds": {"min": [-5, 0, -5], "max": [5, 3, 5]},
            "objects": [
                {
                    "id": "obj_001",
                    "class": "sofa",
                    "position": [2.35, 0.45, 2.65],
                    "dimensions": [2.3, 0.9, 1.1],
                },
                {
                    "id": "obj_002",
                    "class": "coffee_table",
                    "position": [2.5, 0.25, 1.5],
                    "dimensions": [1.2, 0.5, 0.6],
                },
                {
                    "id": "obj_003",
                    "class": "armchair",
                    "position": [0.5, 0.4, 2.0],
                    "dimensions": [0.9, 0.8, 0.9],
                },
                {
                    "id": "obj_004",
                    "class": "armchair",
                    "position": [4.2, 0.4, 2.0],
                    "dimensions": [0.9, 0.8, 0.9],
                },
                {
                    "id": "obj_005",
                    "class": "floor_lamp",
                    "position": [0.3, 0.75, 3.5],
                    "dimensions": [0.3, 1.5, 0.3],
                },
                {
                    "id": "obj_006",
                    "class": "bookshelf",
                    "position": [4.5, 1.0, 4.0],
                    "dimensions": [1.0, 2.0, 0.4],
                },
                {
                    "id": "obj_007",
                    "class": "rug",
                    "position": [2.5, 0.01, 2.0],
                    "dimensions": [3.0, 0.02, 2.0],
                },
                {
                    "id": "obj_008",
                    "class": "tv_stand",
                    "position": [2.5, 0.3, 0.3],
                    "dimensions": [1.8, 0.6, 0.4],
                },
            ],
            "layout": {
                "floor": {"y": 0.0},
                "ceiling": {"y": 2.8},
                "walls": [
                    {"name": "north", "normal": [0, 0, -1], "position": 5.0},
                    {"name": "south", "normal": [0, 0, 1], "position": -0.1},
                    {"name": "east", "normal": [-1, 0, 0], "position": 5.0},
                    {"name": "west", "normal": [1, 0, 0], "position": -0.1},
                ],
                "doors": [{"position": [2.5, 1.0, -0.05], "dimensions": [0.9, 2.1]}],
                "windows": [{"position": [4.9, 1.5, 2.5], "dimensions": [1.5, 1.2]}],
            },
        },
        "answers": [
            {
                "question": "What furniture is in this room?",
                "answer": "I can see a gray sofa, a wooden coffee table, two armchairs, a floor lamp, a bookshelf, a rug, and a TV stand.",
            },
            {
                "question": "How many chairs are there?",
                "answer": "There are 2 armchairs in the room, positioned on either side of the seating area.",
            },
            {
                "question": "What is the color scheme?",
                "answer": "The room features warm neutral tones with beige walls, a gray sofa, and wooden furniture in natural oak finish.",
            },
            {
                "question": "Where is the TV?",
                "answer": "The TV is mounted above or placed on the TV stand, which is positioned against the south wall facing the sofa.",
            },
            {
                "question": "Is there good lighting?",
                "answer": "Yes, there is a floor lamp near the corner and natural light coming from the window on the east wall.",
            },
        ],
    }


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all responses for debugging."""
    request_id = generate_request_id()
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Health endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def get_health() -> HealthResponse:
    """Service health check."""
    return HealthResponse(
        status=HealthStatus.HEALTHY,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
    )


# Session endpoints
@app.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=201,
    tags=["Sessions"],
)
async def create_session(
    images: Annotated[list[UploadFile], File(description="Multi-view images")],
    name: Annotated[str | None, Form(description="Optional session name")] = None,
) -> SessionResponse:
    """Create session with image uploads."""
    if not images:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="NO_IMAGES",
                message="At least one image must be provided",
            ).model_dump(),
        )

    if len(images) > settings.max_images_per_session:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="TOO_MANY_IMAGES",
                message=f"Maximum {settings.max_images_per_session} images allowed",
            ).model_dump(),
        )

    session_id = generate_session_id()
    now = datetime.now(timezone.utc)

    # Create session directory
    session_dir = Path(settings.upload_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded images (in production, would do this async)
    image_count = 0
    for i, image in enumerate(images):
        if image.filename:
            ext = Path(image.filename).suffix or ".jpg"
            filepath = session_dir / f"image_{i:03d}{ext}"
            content = await image.read()
            with open(filepath, "wb") as f:
                f.write(content)
            image_count += 1

    # Store session data
    sessions[session_id] = {
        "session_id": session_id,
        "name": name,
        "status": SessionStatus.CREATED,
        "image_count": image_count,
        "created_at": now,
        "updated_at": now,
        "reconstruction_mode": None,
        "has_reconstruction": False,
        "has_structure": False,
        "has_fallback": False,
        "artifacts": [],
        "instances": [],
        "job_id": None,
        "progress": 0,
        "current_step": None,
        "message": None,
    }

    return SessionResponse(
        session_id=session_id,
        name=name,
        status=SessionStatus.CREATED,
        image_count=image_count,
        created_at=now,
    )


@app.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    tags=["Sessions"],
)
async def get_session(session_id: str) -> SessionDetailResponse:
    """Get session details."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    session = sessions[session_id]
    return SessionDetailResponse(
        session_id=session["session_id"],
        name=session.get("name"),
        status=session["status"],
        image_count=session["image_count"],
        created_at=session["created_at"],
        updated_at=session.get("updated_at"),
        reconstruction_mode=session.get("reconstruction_mode"),
        has_reconstruction=session.get("has_reconstruction", False),
        has_structure=session.get("has_structure", False),
        has_fallback=session.get("has_fallback", False),
        artifact_count=len(session.get("artifacts", [])),
        instance_count=len(session.get("instances", [])),
    )


@app.delete(
    "/sessions/{session_id}",
    status_code=204,
    tags=["Sessions"],
)
async def delete_session(session_id: str):
    """Delete session."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    # Remove session data
    del sessions[session_id]

    # In production, would also clean up files
    return None


@app.get(
    "/sessions/{session_id}/status",
    response_model=SessionStatusResponse,
    tags=["Sessions"],
)
async def get_session_status(session_id: str) -> SessionStatusResponse:
    """Poll processing status."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    session = sessions[session_id]
    started_at = session.get("started_at")
    estimated = None
    if started_at and session["status"] in [
        SessionStatus.PROCESSING,
        SessionStatus.RECONSTRUCTING,
    ]:
        estimated = started_at + timedelta(minutes=5)

    return SessionStatusResponse(
        session_id=session_id,
        status=session["status"],
        current_step=session.get("current_step"),
        progress=session.get("progress", 0),
        message=session.get("message"),
        started_at=started_at,
        estimated_completion=estimated,
        error=session.get("error") if session["status"] == SessionStatus.FAILED else None,
    )


# Reconstruction endpoints
@app.post(
    "/sessions/{session_id}/reconstruct",
    response_model=JobAcceptedResponse,
    status_code=202,
    tags=["Reconstruction"],
)
async def trigger_reconstruction(
    session_id: str,
    request: ReconstructionRequest | None = None,
) -> JobAcceptedResponse:
    """Trigger 3D reconstruction."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    session = sessions[session_id]

    if session["status"] in [SessionStatus.PROCESSING, SessionStatus.RECONSTRUCTING]:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="ALREADY_PROCESSING",
                message="Session is already being processed",
            ).model_dump(),
        )

    mode = request.mode if request else ReconstructionMode.QUICK
    job_id = generate_job_id()
    now = datetime.now(timezone.utc)

    # Update session state
    session["status"] = SessionStatus.RECONSTRUCTING
    session["reconstruction_mode"] = mode
    session["job_id"] = job_id
    session["started_at"] = now
    session["updated_at"] = now
    session["current_step"] = "Initializing reconstruction"
    session["progress"] = 0
    session["message"] = f"Starting {mode.value} reconstruction..."

    # In demo mode, simulate completion
    if settings.demo_mode:
        # Simulate immediate completion for demo
        session["status"] = SessionStatus.READY
        session["has_reconstruction"] = True
        session["progress"] = 100
        session["current_step"] = "Complete"
        session["message"] = "Reconstruction completed successfully"
        session["artifacts"] = [
            {
                "name": "pointcloud.ply",
                "type": ArtifactType.POINTCLOUD,
                "size_bytes": 15234567,
                "created_at": now,
            },
            {
                "name": "poses.json",
                "type": ArtifactType.CAMERA_POSES,
                "size_bytes": 12345,
                "created_at": now,
            },
        ]

    return JobAcceptedResponse(
        job_id=job_id,
        session_id=session_id,
        status=JobStatus.ACCEPTED,
        message=f"{mode.value.title()} reconstruction job queued",
    )


@app.post(
    "/sessions/{session_id}/structure",
    response_model=JobAcceptedResponse,
    status_code=202,
    tags=["Reconstruction"],
)
async def generate_structure(session_id: str) -> JobAcceptedResponse:
    """Generate scene structure."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    session = sessions[session_id]

    if not session.get("has_reconstruction") and not session.get("has_fallback"):
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="RECONSTRUCTION_REQUIRED",
                message="Reconstruction must be completed first",
            ).model_dump(),
        )

    job_id = generate_job_id()
    now = datetime.now(timezone.utc)

    # Update session state
    session["status"] = SessionStatus.STRUCTURING
    session["job_id"] = job_id
    session["updated_at"] = now
    session["current_step"] = "Generating scene structure"
    session["progress"] = 0

    # In demo mode, simulate completion with mock instances
    if settings.demo_mode:
        session["status"] = SessionStatus.READY
        session["has_structure"] = True
        session["progress"] = 100
        session["current_step"] = "Complete"
        session["message"] = "Scene structure generated successfully"

        # Add mock instances
        session["instances"] = [
            Instance(
                id="obj_001",
                class_name="sofa",
                confidence=0.95,
                bounding_box=BoundingBox(
                    min_x=1.2, min_y=0.0, min_z=2.1, max_x=3.5, max_y=0.9, max_z=3.2
                ),
                center=Point3D(x=2.35, y=0.45, z=2.65),
                mask_path="instances/obj_001_mask.png",
            ),
            Instance(
                id="obj_002",
                class_name="coffee_table",
                confidence=0.92,
                bounding_box=BoundingBox(
                    min_x=1.9, min_y=0.0, min_z=1.2, max_x=3.1, max_y=0.5, max_z=1.8
                ),
                center=Point3D(x=2.5, y=0.25, z=1.5),
                mask_path="instances/obj_002_mask.png",
            ),
            Instance(
                id="obj_003",
                class_name="armchair",
                confidence=0.89,
                bounding_box=BoundingBox(
                    min_x=0.05, min_y=0.0, min_z=1.55, max_x=0.95, max_y=0.8, max_z=2.45
                ),
                center=Point3D(x=0.5, y=0.4, z=2.0),
                mask_path="instances/obj_003_mask.png",
            ),
        ]

        # Add scene.json artifact
        session["artifacts"].append(
            {
                "name": "scene.json",
                "type": ArtifactType.SCENE_STRUCTURE,
                "size_bytes": 45678,
                "created_at": now,
            }
        )

    return JobAcceptedResponse(
        job_id=job_id,
        session_id=session_id,
        status=JobStatus.ACCEPTED,
        message="Scene structuring job queued",
    )


# QA endpoint
@app.post(
    "/sessions/{session_id}/ask",
    response_model=AskResponse,
    tags=["QA"],
)
async def ask_question(session_id: str, request: AskRequest) -> AskResponse:
    """Multimodal QA endpoint."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    session = sessions[session_id]

    if session["status"] not in [SessionStatus.READY, SessionStatus.FALLBACK]:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="SESSION_NOT_READY",
                message="Session must be ready before asking questions",
            ).model_dump(),
        )

    # Check for fallback answers
    is_fallback = False
    answer = None

    if session.get("has_fallback") and "fallback_data" in session:
        fallback_answers = session["fallback_data"].get("answers", [])
        prompt_lower = request.prompt.lower()
        for qa in fallback_answers:
            if any(
                word in prompt_lower
                for word in qa["question"].lower().split()
                if len(word) > 3
            ):
                answer = qa["answer"]
                is_fallback = True
                break

    # Generate mock answer if no fallback match
    if not answer:
        if "furniture" in request.prompt.lower() or "object" in request.prompt.lower():
            answer = "I can see a gray sofa, a wooden coffee table, two armchairs, a floor lamp, and a bookshelf against the wall."
        elif "color" in request.prompt.lower() or "style" in request.prompt.lower():
            answer = "The room has warm neutral tones with beige walls, a gray sofa, and wooden furniture in natural oak finish."
        elif "how many" in request.prompt.lower() or "count" in request.prompt.lower():
            answer = "There are 8 objects detected in this scene including furniture and decor items."
        else:
            answer = "This appears to be a modern living room with comfortable seating, good lighting, and functional furniture arrangement."

    # Get referenced objects
    referenced_objects = []
    if request.object_id:
        referenced_objects = [request.object_id]
    elif session.get("instances"):
        referenced_objects = [
            inst.id if isinstance(inst, Instance) else inst["id"]
            for inst in session["instances"][:3]
        ]

    return AskResponse(
        session_id=session_id,
        question=request.prompt,
        answer=answer,
        confidence=0.87,
        referenced_objects=referenced_objects,
        processing_time_ms=1250,
        is_fallback=is_fallback,
    )


# Artifacts endpoints
@app.get(
    "/sessions/{session_id}/artifacts",
    response_model=ArtifactListResponse,
    tags=["Artifacts"],
)
async def list_artifacts(session_id: str) -> ArtifactListResponse:
    """List available artifacts."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    session = sessions[session_id]
    artifacts = []

    for art in session.get("artifacts", []):
        if isinstance(art, dict):
            artifacts.append(
                Artifact(
                    name=art["name"],
                    type=art["type"],
                    size_bytes=art["size_bytes"],
                    created_at=art["created_at"],
                )
            )
        else:
            artifacts.append(art)

    return ArtifactListResponse(
        session_id=session_id,
        artifacts=artifacts,
    )


@app.get(
    "/sessions/{session_id}/artifacts/{name}",
    tags=["Artifacts"],
)
async def download_artifact(session_id: str, name: str):
    """Download artifact."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    session = sessions[session_id]
    artifact_names = [
        (art["name"] if isinstance(art, dict) else art.name)
        for art in session.get("artifacts", [])
    ]

    if name not in artifact_names:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="ARTIFACT_NOT_FOUND",
                message=f"Artifact {name} not found",
            ).model_dump(),
        )

    # Check for actual file
    artifacts_dir = Path(settings.artifacts_dir) / session_id
    filepath = artifacts_dir / name

    if filepath.exists():
        return FileResponse(filepath)

    # Return mock JSON for demo
    if name == "scene.json":
        fallback_data = get_fallback_data()
        return JSONResponse(content=fallback_data["scene"])
    elif name == "poses.json":
        return JSONResponse(
            content={
                "cameras": [
                    {"id": i, "position": [i * 0.5, 1.5, 2.0], "rotation": [0, i * 15, 0]}
                    for i in range(12)
                ]
            }
        )

    # Return empty response for binary files in demo mode
    return JSONResponse(
        content={"message": f"Artifact {name} would be downloaded here"},
        status_code=200,
    )


# Fallback endpoint
@app.post(
    "/sessions/{session_id}/fallback",
    response_model=SessionDetailResponse,
    tags=["Sessions"],
)
async def load_fallback(
    session_id: str,
    request: FallbackRequest | None = None,
) -> SessionDetailResponse:
    """Load precomputed fallback assets."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    scene_type = request.scene_type if request else FallbackSceneType.LIVING_ROOM
    fallback_data = get_fallback_data(scene_type)
    now = datetime.now(timezone.utc)

    session = sessions[session_id]
    session["status"] = SessionStatus.FALLBACK
    session["has_fallback"] = True
    session["has_reconstruction"] = True
    session["has_structure"] = True
    session["updated_at"] = now
    session["fallback_data"] = fallback_data

    # Create instances from fallback data
    session["instances"] = []
    for obj in fallback_data["scene"]["objects"]:
        dims = obj["dimensions"]
        pos = obj["position"]
        session["instances"].append(
            Instance(
                id=obj["id"],
                class_name=obj["class"],
                confidence=0.90,
                bounding_box=BoundingBox(
                    min_x=pos[0] - dims[0] / 2,
                    min_y=pos[1] - dims[1] / 2,
                    min_z=pos[2] - dims[2] / 2,
                    max_x=pos[0] + dims[0] / 2,
                    max_y=pos[1] + dims[1] / 2,
                    max_z=pos[2] + dims[2] / 2,
                ),
                center=Point3D(x=pos[0], y=pos[1], z=pos[2]),
            )
        )

    # Add fallback artifacts
    session["artifacts"] = [
        {
            "name": "pointcloud.ply",
            "type": ArtifactType.POINTCLOUD,
            "size_bytes": 15234567,
            "created_at": now,
        },
        {
            "name": "scene.json",
            "type": ArtifactType.SCENE_STRUCTURE,
            "size_bytes": 45678,
            "created_at": now,
        },
        {
            "name": "poses.json",
            "type": ArtifactType.CAMERA_POSES,
            "size_bytes": 12345,
            "created_at": now,
        },
    ]

    return SessionDetailResponse(
        session_id=session_id,
        name=session.get("name"),
        status=SessionStatus.FALLBACK,
        image_count=session["image_count"],
        created_at=session["created_at"],
        updated_at=now,
        reconstruction_mode=session.get("reconstruction_mode"),
        has_reconstruction=True,
        has_structure=True,
        has_fallback=True,
        artifact_count=len(session["artifacts"]),
        instance_count=len(session["instances"]),
    )


# Instances endpoint
@app.get(
    "/instances/{session_id}",
    response_model=InstanceListResponse,
    tags=["Instances"],
)
async def list_instances(session_id: str) -> InstanceListResponse:
    """List detected instances."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="SESSION_NOT_FOUND",
                message=f"Session {session_id} not found",
            ).model_dump(),
        )

    session = sessions[session_id]
    instances = []

    for inst in session.get("instances", []):
        if isinstance(inst, Instance):
            instances.append(inst)
        else:
            instances.append(
                Instance(
                    id=inst["id"],
                    class_name=inst["class_name"],
                    confidence=inst.get("confidence"),
                    bounding_box=BoundingBox(**inst["bounding_box"]),
                    center=Point3D(**inst["center"]) if inst.get("center") else None,
                    mask_path=inst.get("mask_path"),
                )
            )

    return InstanceListResponse(
        session_id=session_id,
        instance_count=len(instances),
        instances=instances,
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    request_id = getattr(request.state, "request_id", None)

    if isinstance(exc.detail, dict):
        content = exc.detail
        content["request_id"] = request_id
    else:
        content = {
            "error": "HTTP_ERROR",
            "message": str(exc.detail),
            "request_id": request_id,
        }

    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    request_id = getattr(request.state, "request_id", None)

    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An internal error occurred",
            "request_id": request_id,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
