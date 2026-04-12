# Space Object Detection — ML Pipeline

End-to-end ML pipeline for detecting objects in astronomical sky survey images
using YOLOS-small (Vision Transformer) and MLflow for experiment tracking.

## Setup

```bash
uv sync
```

## Run the pipeline

```bash
uv run python -m src.pipeline --n-regions 20 --confidence-threshold 0.4
```

## View experiments

```bash
uv run mlflow ui --port 5000
```

## Serve the model

```bash
uv run mlflow models serve -m "models:/space-detector/1" --port 5001 --no-conda
```

## Run inference

```bash
uv run python -m src.inference --image data/raw/field_0000.jpg
```

## Run tests

```bash
uv run pytest tests/ -v
```

## Pipeline parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-regions` | 20 | Number of sky regions to download from SDSS |
| `--radius-deg` | 0.05 | Search radius per region (degrees) |
| `--scale` | 0.2 | Image scale in arcsec/pixel |
| `--confidence-threshold` | 0.4 | Minimum detection confidence |
