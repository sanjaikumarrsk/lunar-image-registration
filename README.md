# Lunar Image Registration

An upload-driven lunar image registration demo with a FastAPI backend, React/Vite frontend, MongoDB persistence, and a geometric LoFTR matching pipeline.

## What is included

- `backend/`: FastAPI API, MongoDB store, image preprocessing, matching, refinement, validation, and registration pipeline.
- `frontend/`: React/Vite demo UI.
- `models/finetuned_loftr/kaguya_geometric/best_model.pth`: checkpoint used by the default LoFTR pipeline.
- `configs/loftr.yaml`: LoFTR training configuration.
- `requirements.txt` and `frontend/package-lock.json`: runtime dependency manifests.

Raw lunar datasets, downloaded DEMs, generated run artifacts, caches, and duplicate checkpoints are intentionally excluded. The web demo accepts source and reference images as uploads, so the repository can run without a bundled dataset.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

cd frontend
npm install
```

PyTorch may need a platform-specific installation command for GPU support. The pipeline automatically uses CUDA when available and otherwise runs on CPU.

## MongoDB

Start MongoDB locally or provide these environment variables before starting the backend:

```powershell
$env:MONGO_URI = "mongodb://localhost:27017"
$env:MONGO_DATABASE = "lunar"
$env:MONGO_COLLECTION = "lunarimage"
```

The API remains available when MongoDB is unavailable, but registration history persistence and history endpoints require a reachable MongoDB instance.

## Run the demo

From the repository root, start the backend:

```powershell
python -m uvicorn backend.api.api_server:app --reload --port 8000
```

In a second terminal, start the frontend:

```powershell
cd frontend
npm run dev
```

Open the Vite URL shown in the terminal, upload two lunar images, and select **Run Registration**. The API stores generated artifacts under `runs/`; that directory is ignored by Git.

For a direct pipeline run without the UI:

```powershell
python backend/run_pipeline.py --source path\to\source.png --reference path\to\reference.png --method loftr
```

Use `--method sift` when a LoFTR checkpoint is not desired. Outputs are written to the ignored `results/` directory unless `--output-dir` is supplied.

## API checks

```text
GET  /api/health
POST /api/register   (multipart fields: source, reference)
```

Set `frontend/.env` from `frontend/.env.example` if the backend runs at a different URL.
