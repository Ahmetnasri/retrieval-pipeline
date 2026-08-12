# Image Retrieval Pipeline

A GPU-accelerated image retrieval and visual matching pipeline for finding identical or visually similar images in large image collections.

The project combines **global image embeddings**, **vector similarity search**, and **local feature matching** in a multi-stage retrieval pipeline. It is designed as a reusable foundation for applications such as visual search, image re-identification, duplicate detection, industrial inspection, product matching, and fine-grained instance retrieval.

## Overview

The retrieval process is divided into multiple stages:

1. **Global feature extraction** – images are encoded into compact feature vectors using an SSCD-based embedding model.
2. **Vector search** – embeddings are stored and queried in **Qdrant** using cosine similarity.
3. **Candidate refinement** – a second embedding stage reduces the candidate set for more focused matching.
4. **Local feature matching** – **SuperPoint** or **DISK** features are matched using **LightGlue** for more precise verification.
5. **Rotation handling** – optional rotated queries can be used to improve robustness when image orientation is not fixed.

This architecture combines the scalability of vector search with the accuracy of local feature matching.

## Features

* Multi-stage image retrieval
* SSCD-based global image embeddings
* Qdrant vector database integration
* SuperPoint and DISK local feature extraction
* LightGlue feature matching
* Optional rotation-aware retrieval
* GPU-accelerated inference with PyTorch
* FastAPI REST API
* Configurable retrieval limits, thresholds, models, and storage paths
* Docker Compose setup for Qdrant

## Pipeline

```text
                         Query Image
                              |
                              v
                    Global Embedding Model
                              |
                              v
                      Qdrant Vector Search
                              |
                       Candidate Images
                              |
                              v
                    Second Retrieval Stage
                              |
                              v
                     Refined Candidates
                              |
                              v
                  Local Feature Extraction
                    SuperPoint / DISK
                              |
                              v
                      LightGlue Matching
                              |
                              v
                       Matching Scores
```

## Project Structure

```text
retrieval-pipeline/
├── .env_example
├── .gitignore
├── docker-compose.yml
├── README.md
└── src/
    ├── app.py
    ├── main.py
    ├── lightweight_model.py
    ├── main_decision_maker.py
    ├── retrieve.py
    ├── superpoint_modified.py
    └── helpers/
        ├── __init__.py
        └── config.py
```

## Main Components

### Global Embedding Retrieval

The first retrieval stages use SSCD-based image embeddings. The generated vectors are stored in Qdrant and compared using cosine similarity to quickly retrieve the most relevant candidates from a larger database.

### Local Feature Matching

For more precise verification, the pipeline supports **SuperPoint** and **DISK** local feature extractors together with **LightGlue**. This stage compares local structures between the query and the candidates returned by the vector search.

### Two-Stage Retrieval

The pipeline supports two Qdrant collections with independent embedding models and vector dimensionalities. The first stage performs broad candidate retrieval, while the second stage searches within the candidates returned by the first stage.

## Requirements

The project is designed to run with a CUDA-capable GPU.

Core dependencies include:

* Python
* PyTorch
* TorchVision
* FastAPI
* Uvicorn
* Qdrant Client
* LightGlue
* Safetensors
* Pillow
* NumPy
* Pydantic Settings
* aiofiles
* python-multipart

The SSCD model weights used by the embedding stages must be available locally at the paths configured in the environment file.

## Configuration

Create a local environment file from the provided example:

```bash
cp .env_example .env
```

The main configuration options include:

```env
DEVICE="cuda"

APP_HOST="0.0.0.0"
APP_PORT=8003
WORKERS=1

QDRANT_HOST="localhost"
DB_COLLECTION_NAME_PHASE1="image_retrieval_phase1"
DB_COLLECTION_NAME_PHASE2="image_retrieval_phase2"
DB_LIMIT_PHASE1=500
DB_LIMIT_PHASE2=500
DB_DIMENSIONALITY_PHASE1=1024
DB_DIMENSIONALITY_PHASE2=512

EMBEDDING_MODEL_PHASE1="sscd"
EMBEDDING_MODEL_PHASE2="sscd"
EMBEDDING_MODEL_PATH_PHASE1="models/model_phase1.torchscript.pt"
EMBEDDING_MODEL_PATH_PHASE2="models/model_phase2.torchscript.pt"

EXTRACTOR_MODEL="superpoint"
NUM_PATCH=1
DISK_KEYPOINTS=256
THRESHOLDS=[0.85, 0.8, 0.7]
```

Collection names, model paths, database limits, storage paths, and feature-matching settings can be adapted to the target application.

## Run Qdrant

Qdrant can be started using the included Docker Compose configuration:

```bash
docker compose up -d
```

By default, Qdrant is exposed on port `6333`.

Configure the volume path in `docker-compose.yml` according to the local environment where the database should be stored.

## Run the API

From the `src` directory:

```bash
cd src
python main.py
```

The API runs by default at:

```text
http://localhost:8003
```

FastAPI documentation is available at:

```text
http://localhost:8003/docs
```

## API Endpoints

### Health Check

```http
GET /health
```

### Add an Embedding

```http
POST /add_embedding_to_database
```

Adds the embedding of an uploaded image to one of the configured Qdrant collections.

Example:

```bash
curl -X POST \
  "http://localhost:8003/add_embedding_to_database?phase=1" \
  -F "image_file=@image.jpg"
```

### Vector Search

```http
POST /query_lightweight
```

Extracts an embedding from the uploaded query image and performs vector similarity search.

Example:

```bash
curl -X POST \
  "http://localhost:8003/query_lightweight?phase=1" \
  -F "image_file=@query.jpg"
```

### Local Feature Matching

```http
POST /query_with_lightglue
```

Runs local feature extraction and LightGlue matching against the configured local feature database.

### Full Retrieval Pipeline

```http
POST /full_query
```

Runs the complete retrieval workflow:

* Phase 1 global embedding search
* Phase 2 candidate refinement
* Local feature extraction
* LightGlue verification
* Optional rotation-aware matching

Example:

```bash
curl -X POST \
  "http://localhost:8003/full_query" \
  -F "image_file_phase1=@query_stage1.jpg" \
  -F "image_file_phase2=@query_stage2.jpg" \
  -F "rotate_embeddding_phase1=false" \
  -F "rotate_embeddding_phase2=false" \
  -F "rotate_lightglue=false"
```

The response contains the retrieval results from both Qdrant stages together with the local feature matching scores.

## Use Cases

The pipeline can be adapted to a variety of image retrieval tasks, including:

* Visual similarity search
* Image re-identification
* Object-instance retrieval
* Duplicate and near-duplicate detection
* Product or component matching
* Industrial visual inspection
* Reference-image lookup
* Fine-grained visual matching
* Large-scale image database search

## Customization

The main retrieval components are modular and can be adapted independently. Depending on the application, it is possible to change:

* Global embedding models
* Embedding dimensionality
* Qdrant collection configuration
* Candidate retrieval limits
* Local feature extractor
* Number of keypoints
* LightGlue matching strategy
* Matching thresholds
* Rotation handling
* Database and storage paths

## License

This repository is not licensed for reuse, modification, redistribution, or commercial use.

**All rights reserved.**
