# 3D Reconstruction + Multimodal QA Backend

A demo-focused project for multi-view 3D reconstruction with instance segmentation and multimodal question answering over reconstructed scenes.

## 🎯 Features

- **Multi-view 3D Reconstruction**: Generate point clouds and camera poses from image sets
- **Instance Segmentation**: Detect and segment objects in the scene
- **Multimodal QA**: Ask questions about the scene using LMM (Qwen2-VL)
- **Bulletproof Demo**: Precomputed fallback assets ensure demo reliability
- **Interactive Viewer**: Three.js-based point cloud and bounding box visualization

## 📁 Project Structure

```
├── api/                    # FastAPI backend
│   ├── __init__.py
│   ├── main.py             # FastAPI application with all endpoints
│   ├── models.py           # Pydantic models matching OpenAPI spec
│   ├── tasks.py            # Celery task stubs
│   ├── config.py           # Configuration management
│   └── requirements.txt    # API dependencies
├── notebooks/
│   └── demo_pipeline.ipynb # Kaggle notebook for GPU pipeline
├── viewer/                 
│   └── index.html          # Three.js point cloud viewer
├── fallback/               
│   └── .gitkeep            # Precomputed demo assets
├── specs/
│   └── openapi.yaml        # OpenAPI 3.1 specification
├── docs/
│   └── backend_overview.md # Architecture documentation
├── docker-compose.yml      # Redis + API setup
├── Dockerfile              # API container definition
├── requirements.txt        # Full dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/3D-Scan-to-Smart-Homes.git
   cd 3D-Scan-to-Smart-Homes
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r api/requirements.txt
   ```

4. **Run the API server**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

5. **Open the API docs**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Using the Viewer

1. Open `viewer/index.html` in a browser
2. Click "Load Demo Scene" to see a sample visualization
3. Use mouse to rotate (drag), zoom (scroll), and pan (right-click drag)

## 📓 Kaggle Notebook

The notebook at `notebooks/demo_pipeline.ipynb` runs the full pipeline on Kaggle/Colab:

1. **Upload to Kaggle** and enable GPU (T4 recommended)
2. **Run all cells** - the notebook will:
   - Install dependencies
   - Generate mock reconstruction data
   - Create scene structure
   - Test multimodal QA
   - Export demo assets as `demo_ready.zip`

### GPU Requirements
- **T4 (16GB)**: Use Qwen2-VL-2B with 4-bit quantization
- **P100 (16GB)**: Use Qwen2-VL-7B with 4-bit quantization

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/sessions` | Create session with image uploads |
| `GET` | `/sessions/{id}` | Get session details |
| `DELETE` | `/sessions/{id}` | Delete session |
| `GET` | `/sessions/{id}/status` | Poll processing status |
| `POST` | `/sessions/{id}/reconstruct` | Trigger 3D reconstruction |
| `POST` | `/sessions/{id}/structure` | Generate scene structure |
| `POST` | `/sessions/{id}/ask` | Multimodal QA |
| `GET` | `/sessions/{id}/artifacts` | List artifacts |
| `GET` | `/sessions/{id}/artifacts/{name}` | Download artifact |
| `POST` | `/sessions/{id}/fallback` | Load fallback assets |
| `GET` | `/instances/{id}` | List detected instances |

See `specs/openapi.yaml` for complete API specification.

## 🎭 Demo Workflow

### Normal Demo Flow
1. Create session with uploaded images
2. Trigger reconstruction (quick mode)
3. Generate scene structure
4. Ask questions about the scene
5. Download artifacts (point cloud, scene.json)

### Fallback Procedure (If Live Processing Fails)
1. Create session with any images
2. Call `/sessions/{id}/fallback` to load precomputed assets
3. Proceed with QA using fallback data
4. Demo continues without interruption

## 🛠 Configuration

Environment variables (see `api/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_DEBUG` | `false` | Enable debug mode |
| `APP_DEMO_MODE` | `true` | Enable demo mode with mocks |
| `APP_REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `APP_LMM_MODEL_NAME` | `Qwen/Qwen2-VL-2B-Instruct` | LMM model |
| `APP_LMM_QUANTIZATION` | `4bit` | Quantization level |

## 📚 Documentation

- [Backend Architecture](docs/backend_overview.md) - Detailed system design
- [OpenAPI Spec](specs/openapi.yaml) - API specification

## 🧪 Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black api/
ruff check api/
```

### Building Docker Image
```bash
docker build -t 3d-recon-api .
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details