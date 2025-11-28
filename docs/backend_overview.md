# Backend Architecture Overview

This document provides a comprehensive overview of the 3D Reconstruction + Multimodal QA backend system.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │   Web App    │  │   Viewer     │  │   Notebook   │                       │
│  │  (Frontend)  │  │  (Three.js)  │  │  (Kaggle)    │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         FastAPI Application                          │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │    │
│  │  │   Session   │ │ Reconstruct │ │     QA      │ │  Artifacts  │   │    │
│  │  │  Endpoints  │ │  Endpoints  │ │  Endpoints  │ │  Endpoints  │   │    │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌────────────────────────┐ ┌─────────────┐ ┌─────────────────────────────────┐
│     TASK LAYER         │ │   STORAGE   │ │         COMPUTE LAYER           │
├────────────────────────┤ ├─────────────┤ ├─────────────────────────────────┤
│  ┌──────────────────┐  │ │   Redis     │ │  ┌───────────────────────────┐  │
│  │  Celery Worker   │  │ │  (Broker)   │ │  │   Kaggle/Colab Notebook   │  │
│  │  ┌────────────┐  │  │ │             │ │  │  ┌─────────────────────┐  │  │
│  │  │   Recon    │  │  │ │  ┌───────┐  │ │  │  │   Reconstruction    │  │  │
│  │  │   Task     │  │  │ │  │Session│  │ │  │  │   (IGGT/DUSt3R)     │  │  │
│  │  ├────────────┤  │  │ │  │ Store │  │ │  │  ├─────────────────────┤  │  │
│  │  │   Scene    │  │  │ │  └───────┘  │ │  │  │   Segmentation      │  │  │
│  │  │   Task     │  │  │ │             │ │  │  │   (SAM)             │  │  │
│  │  ├────────────┤  │  │ │  ┌───────┐  │ │  │  ├─────────────────────┤  │  │
│  │  │   LMM      │  │  │ │  │ File  │  │ │  │  │   LMM Inference     │  │  │
│  │  │   Task     │  │  │ │  │Storage│  │ │  │  │   (Qwen2-VL)        │  │  │
│  │  └────────────┘  │  │ │  └───────┘  │ │  │  └─────────────────────┘  │  │
│  └──────────────────┘  │ │             │ │  └───────────────────────────┘  │
└────────────────────────┘ └─────────────┘ └─────────────────────────────────┘
```

## Components

### 1. API Layer (FastAPI)

The API layer handles all HTTP requests and provides a RESTful interface.

**Key Files:**
- `api/main.py` - FastAPI application with route handlers
- `api/models.py` - Pydantic request/response models
- `api/config.py` - Configuration management

**Features:**
- Async request handling
- Request ID tracking for debugging
- CORS support for frontend integration
- Automatic OpenAPI documentation

### 2. Task Layer (Celery)

Background task processing for compute-intensive operations.

**Tasks:**
- `run_reconstruction` - 3D point cloud generation
- `run_scene_structuring` - Instance segmentation and scene analysis
- `run_lmm_inference` - Multimodal QA inference

**Configuration:**
- Redis as message broker
- JSON serialization
- Task progress tracking

### 3. Storage Layer

**Session Storage (In-Memory/Redis):**
- Session metadata
- Processing status
- Detected instances

**File Storage:**
- Uploaded images (`uploads/{session_id}/`)
- Generated artifacts (`artifacts/{session_id}/`)
- Fallback assets (`fallback/`)

### 4. Compute Layer (Kaggle/Colab)

GPU-accelerated processing for ML models.

**Components:**
- 3D Reconstruction (IGGT or DUSt3R)
- Instance Segmentation (SAM)
- Multimodal LLM (Qwen2-VL)

## Data Flow

### Session Creation Flow
```
Client                  API                     Storage
  │                      │                         │
  │ POST /sessions       │                         │
  │ (with images)        │                         │
  │─────────────────────>│                         │
  │                      │ Save images             │
  │                      │────────────────────────>│
  │                      │                         │
  │                      │ Create session record   │
  │                      │────────────────────────>│
  │                      │                         │
  │  201 SessionResponse │                         │
  │<─────────────────────│                         │
```

### Reconstruction Flow
```
Client              API              Celery           Compute
  │                  │                  │                │
  │ POST /reconstruct│                  │                │
  │─────────────────>│                  │                │
  │                  │ Queue task       │                │
  │                  │─────────────────>│                │
  │  202 JobAccepted │                  │                │
  │<─────────────────│                  │                │
  │                  │                  │ Process        │
  │                  │                  │───────────────>│
  │ GET /status      │                  │                │
  │─────────────────>│                  │                │
  │  StatusResponse  │                  │                │
  │<─────────────────│                  │                │
  │                  │                  │    Result      │
  │                  │                  │<───────────────│
  │ GET /status      │                  │                │
  │─────────────────>│                  │                │
  │  status: ready   │                  │                │
  │<─────────────────│                  │                │
```

### QA Flow
```
Client              API              LMM Service
  │                  │                    │
  │ POST /ask        │                    │
  │ {prompt: "..."}  │                    │
  │─────────────────>│                    │
  │                  │ Load scene data    │
  │                  │ Prepare context    │
  │                  │ Invoke LMM         │
  │                  │───────────────────>│
  │                  │     Answer         │
  │                  │<───────────────────│
  │   AskResponse    │                    │
  │<─────────────────│                    │
```

## Data Contracts

### Session Object
```json
{
  "session_id": "ses_abc123def456",
  "name": "Living Room Scan",
  "status": "ready",
  "image_count": 12,
  "created_at": "2025-01-15T10:30:00Z",
  "has_reconstruction": true,
  "has_structure": true,
  "artifact_count": 3,
  "instance_count": 8
}
```

### Scene Structure (scene.json)
```json
{
  "name": "Reconstructed Living Room",
  "bounds": {
    "min": [-5, 0, -5],
    "max": [5, 3, 5]
  },
  "objects": [
    {
      "id": "obj_001",
      "class": "sofa",
      "position": [2.35, 0.45, 2.65],
      "dimensions": [2.3, 0.9, 1.1],
      "bounding_box": {
        "min": [1.2, 0, 2.1],
        "max": [3.5, 0.9, 3.2]
      },
      "confidence": 0.95
    }
  ],
  "layout": {
    "floor": {"y": 0.0},
    "ceiling": {"y": 2.8},
    "walls": [...],
    "doors": [...],
    "windows": [...]
  }
}
```

### Instance Object
```json
{
  "id": "obj_001",
  "class_name": "sofa",
  "confidence": 0.95,
  "bounding_box": {
    "min_x": 1.2, "min_y": 0.0, "min_z": 2.1,
    "max_x": 3.5, "max_y": 0.9, "max_z": 3.2
  },
  "center": {"x": 2.35, "y": 0.45, "z": 2.65},
  "mask_path": "instances/obj_001_mask.png"
}
```

## Compute Guidance

### GPU Requirements

| Model | VRAM | Quantization | Speed |
|-------|------|--------------|-------|
| Qwen2-VL-2B | 8GB | 4-bit | Fast |
| Qwen2-VL-7B | 16GB | 4-bit | Medium |
| Qwen2-VL-7B | 24GB | 8-bit | Better |

### Recommended Configurations

**Demo (T4 GPU - 16GB)**
```python
config = {
    "lmm_model": "Qwen/Qwen2-VL-2B-Instruct",
    "quantization": "4bit",
    "reconstruction_mode": "quick",
    "max_images": 20
}
```

**Production (A100 GPU - 40GB)**
```python
config = {
    "lmm_model": "Qwen/Qwen2-VL-7B-Instruct",
    "quantization": "8bit",
    "reconstruction_mode": "full",
    "max_images": 100
}
```

## Reliability Strategies

### 1. Fallback Assets

Pre-computed demo assets ensure demo success even if:
- GPU processing fails
- Model loading errors
- Timeout issues

**Fallback Trigger:**
```python
POST /sessions/{session_id}/fallback
{
  "scene_type": "living_room"
}
```

### 2. Graceful Degradation

```python
def ask_question(prompt, model, scene):
    try:
        # Try LMM inference
        return lmm_inference(prompt, model)
    except Exception:
        # Fall back to keyword matching
        return fallback_response(prompt, scene)
```

### 3. Status Polling

Long-running operations use status polling:
1. Submit job → get `job_id`
2. Poll `/status` until complete
3. Handle timeout with fallback

### 4. Request ID Tracking

Every request gets a unique ID for debugging:
```
X-Request-ID: req_abc123
```

## Security Considerations

### Input Validation
- File type validation for uploads
- Size limits on images
- Prompt length limits for QA

### Rate Limiting
- Per-session limits on requests
- API-wide rate limiting recommended

### Data Isolation
- Sessions are isolated by ID
- Users can only access their sessions

## Deployment Options

### 1. Local Development
```bash
uvicorn api.main:app --reload
```

### 2. Docker Compose
```bash
docker-compose up
```

### 3. Cloud Deployment
- API: Cloud Run, App Engine, ECS
- Worker: Compute Engine, EC2 with GPU
- Storage: Cloud Storage, S3

## Monitoring

### Health Check
```bash
GET /health
```

### Metrics to Track
- Request latency
- GPU memory usage
- Task queue length
- Error rates

### Logging
- Request/response logging
- Task execution logs
- Error tracing with request IDs
