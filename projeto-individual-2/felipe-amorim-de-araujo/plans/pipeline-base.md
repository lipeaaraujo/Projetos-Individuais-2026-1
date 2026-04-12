# Space Object Detection Pipeline — Base Implementation Plan

> **For the agent:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end ML pipeline that downloads SDSS sky images, runs YOLOS-small inference to detect objects as bounding boxes, tracks everything with MLflow, and serves the registered model via `mlflow models serve`.

**Architecture:** Parameterized SDSS ingestion → arcsinh preprocessing → YOLOS-small inference (bounding boxes + confidence only, no class remapping) → guardrails (input + output validation) → MLflow tracking + model registry → deploy via `mlflow models serve`.

**Tech Stack:** Python 3.13, uv, `transformers` (YOLOS-small), `astroquery` + `astropy` (SDSS), `mlflow`, `Pillow`, `torch`, `requests`, `pytest`

---

## Project structure to build

```
felipe-amorim-de-araujo/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   └── preprocess.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── guardrails.py
│   ├── pipeline.py
│   └── inference.py
├── tests/
│   ├── data/
│   │   ├── test_ingest.py
│   │   └── test_preprocess.py
│   ├── model/
│   │   ├── test_detector.py
│   │   └── test_guardrails.py
│   └── test_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── mlruns/
├── pyproject.toml   (already exists — update dependencies)
└── README.md
```

---

## Task 1: Project dependencies and structure

**Files:**
- Modify: `pyproject.toml`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/model/__init__.py`
- Create: `tests/data/`, `tests/model/` directories

**Step 1: Update pyproject.toml with all dependencies**

```toml
[project]
name = "felipe-amorim-de-araujo"
version = "0.1.0"
description = "Space object detection pipeline with YOLOS-small and MLflow"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "mlflow>=2.13.0",
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "Pillow>=10.0.0",
    "astroquery>=0.4.7",
    "astropy>=6.0.0",
    "requests>=2.31.0",
    "numpy>=1.26.0",
    "pytest>=8.0.0",
    "timm>=0.9.0",
]
```

**Step 2: Install dependencies**

```bash
cd /home/felipe/Workspace/Projetos-Individuais-2026-1/projeto-individual-2/felipe-amorim-de-araujo
uv sync
```

Expected: all packages install without error.

**Step 3: Create directory structure and empty `__init__.py` files**

```bash
mkdir -p src/data src/model tests/data tests/model data/raw data/processed mlruns
touch src/__init__.py src/data/__init__.py src/model/__init__.py
touch tests/__init__.py tests/data/__init__.py tests/model/__init__.py
```

**Step 4: Commit**

```bash
git add pyproject.toml src/ tests/ data/ mlruns/
git commit -m "chore: scaffold project structure and dependencies"
```

---

## Task 2: SDSS data ingestion (`src/data/ingest.py`)

**Files:**
- Create: `src/data/ingest.py`
- Create: `tests/data/test_ingest.py`

**Step 1: Write the failing tests**

```python
# tests/data/test_ingest.py
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import pytest
from src.data.ingest import SAMPLE_REGIONS, build_dataset, download_cutout


def test_sample_regions_not_empty():
    assert len(SAMPLE_REGIONS) > 0
    for ra, dec in SAMPLE_REGIONS:
        assert 0 <= ra <= 360
        assert -90 <= dec <= 90


def test_download_cutout_creates_file(tmp_path):
    """Mock HTTP call — verify file is written."""
    from PIL import Image
    import io

    fake_img = Image.new("RGB", (640, 640), color=(10, 10, 10))
    buf = io.BytesIO()
    fake_img.save(buf, format="JPEG")
    buf.seek(0)

    mock_response = MagicMock()
    mock_response.content = buf.read()
    mock_response.raise_for_status = MagicMock()

    with patch("src.data.ingest.requests.get", return_value=mock_response):
        out = download_cutout(180.0, 0.0, tmp_path / "test.jpg")

    assert out.exists()
    assert out.suffix == ".jpg"


def test_build_dataset_skips_empty_regions(tmp_path):
    """If SDSS returns no objects, skip region gracefully."""
    with patch("src.data.ingest.query_region", return_value=pd.DataFrame()), \
         patch("src.data.ingest.download_cutout") as mock_dl:
        build_dataset(tmp_path, n_regions=2)
        mock_dl.assert_not_called()
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data/test_ingest.py -v
```

Expected: `ImportError` — `src.data.ingest` does not exist yet.

**Step 3: Implement `src/data/ingest.py`**

```python
# src/data/ingest.py
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u

SDSS_CUTOUT_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"

CLASS_MAP = {3: "galaxy", 6: "star", 5: "quasar"}

# Curated sky regions: mix of galaxy-rich and sparse fields
SAMPLE_REGIONS = [
    (180.0,   0.0),   # equatorial field
    (210.0,  54.0),   # near Virgo cluster (galaxy-rich)
    (130.0,  20.0),
    (240.0,  30.0),
    (150.0,  10.0),
    (200.0,  -5.0),
    (170.0,  45.0),
    (190.0,  25.0),
    (220.0,  15.0),
    (160.0,  35.0),
    (230.0,   5.0),
    (140.0,  50.0),
    (250.0,  20.0),
    (175.0,  -2.0),
    (205.0,  60.0),
    (185.0,  40.0),
    (215.0,  -8.0),
    (155.0,  28.0),
    (245.0,  42.0),
    (195.0,  12.0),
]


def query_region(ra: float, dec: float, radius_deg: float = 0.05) -> pd.DataFrame:
    """Query SDSS PhotoObj catalog for objects in a sky region."""
    coord = SkyCoord(ra=ra, dec=dec, unit=u.deg)
    result = SDSS.query_region(
        coordinates=coord,
        radius=radius_deg * u.deg,
        photoobj_fields=["objID", "ra", "dec", "type", "petroRad_r", "psfMag_r", "flags"],
        data_release=17,
    )
    if result is None:
        return pd.DataFrame()

    df = result.to_pandas()
    df = df[df["type"].isin(CLASS_MAP.keys())].copy()
    df = df[df["psfMag_r"] < 22.0]
    df = df[df["flags"] == 0]
    df["class_name"] = df["type"].map(CLASS_MAP)
    return df.reset_index(drop=True)


def download_cutout(
    ra: float,
    dec: float,
    output_path: Path,
    scale: float = 0.2,
    width: int = 640,
    height: int = 640,
) -> Path:
    """Download a JPEG sky cutout centered on (ra, dec)."""
    params = {"ra": ra, "dec": dec, "scale": scale, "width": width, "height": height, "opt": ""}
    response = requests.get(SDSS_CUTOUT_URL, params=params, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img.save(output_path)
    return output_path


def build_dataset(
    output_dir: Path,
    n_regions: int = 20,
    radius_deg: float = 0.05,
    scale: float = 0.2,
) -> dict:
    """Download images for N sky regions. Returns ingestion stats."""
    images_dir = output_dir / "raw"
    images_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0

    for i, (ra, dec) in enumerate(SAMPLE_REGIONS[:n_regions]):
        objects = query_region(ra, dec, radius_deg=radius_deg)
        if objects.empty:
            skipped += 1
            continue

        img_path = images_dir / f"field_{i:04d}.jpg"
        download_cutout(ra, dec, img_path, scale=scale)
        downloaded += 1

    return {"downloaded": downloaded, "skipped": skipped}
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data/test_ingest.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add src/data/ingest.py tests/data/test_ingest.py
git commit -m "feat: SDSS data ingestion with parameterized regions"
```

---

## Task 3: Image preprocessing (`src/data/preprocess.py`)

**Files:**
- Create: `src/data/preprocess.py`
- Create: `tests/data/test_preprocess.py`

**Step 1: Write the failing tests**

```python
# tests/data/test_preprocess.py
import numpy as np
import pytest
from PIL import Image
from src.data.preprocess import arcsinh_stretch, preprocess_image


def test_arcsinh_stretch_output_range():
    """Output should be uint8 image with values in [0, 255]."""
    arr = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    result = arcsinh_stretch(arr)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_arcsinh_stretch_dark_image_not_clipped():
    """Very dark images should still produce non-zero output."""
    arr = np.ones((100, 100), dtype=np.float32) * 10
    result = arcsinh_stretch(arr)
    assert result.max() > 0


def test_preprocess_image_returns_pil():
    """preprocess_image should return a PIL RGB image."""
    img = Image.new("RGB", (640, 640), color=(20, 20, 30))
    result = preprocess_image(img)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"


def test_preprocess_image_preserves_content():
    """Non-blank image should not become blank after preprocessing."""
    arr = np.random.randint(10, 200, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    result = preprocess_image(img)
    result_arr = np.array(result)
    assert result_arr.mean() > 0
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data/test_preprocess.py -v
```

Expected: `ImportError` — module does not exist.

**Step 3: Implement `src/data/preprocess.py`**

```python
# src/data/preprocess.py
import numpy as np
from PIL import Image


def arcsinh_stretch(data: np.ndarray, scale_percentile: float = 99.0) -> np.uint8:
    """
    Apply arcsinh stretch to compress astronomical image dynamic range.
    Standard technique: compresses bright stars while preserving faint sources.
    """
    data = data.astype(np.float32)
    p99 = np.percentile(data, scale_percentile)
    if p99 == 0:
        p99 = 1.0  # avoid division by zero on blank images
    stretched = np.arcsinh(data / p99)
    max_val = np.arcsinh(1.0)
    normalized = np.clip(stretched / max_val, 0, 1)
    return (normalized * 255).astype(np.uint8)


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocess a PIL RGB image for YOLOS input:
    - Apply arcsinh stretch per channel to handle astronomical dynamic range
    - Return PIL RGB image (YolosImageProcessor handles final resize)
    """
    arr = np.array(img)  # shape: (H, W, 3)
    stretched = np.stack([arcsinh_stretch(arr[:, :, c]) for c in range(3)], axis=2)
    return Image.fromarray(stretched, mode="RGB")
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data/test_preprocess.py -v
```

Expected: all 4 tests PASS.

**Step 5: Commit**

```bash
git add src/data/preprocess.py tests/data/test_preprocess.py
git commit -m "feat: arcsinh preprocessing for astronomical images"
```

---

## Task 4: Guardrails (`src/model/guardrails.py`)

**Files:**
- Create: `src/model/guardrails.py`
- Create: `tests/model/test_guardrails.py`

**Step 1: Write the failing tests**

```python
# tests/model/test_guardrails.py
import pytest
import numpy as np
from PIL import Image
from src.model.guardrails import (
    validate_input,
    validate_output,
    GuardrailError,
)


def make_image(w=640, h=640, mode="RGB", value=30):
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


# --- Input guardrails ---

def test_valid_image_passes():
    img = make_image()
    validate_input(img)  # should not raise


def test_rejects_non_rgb():
    img = Image.new("L", (640, 640))  # grayscale
    with pytest.raises(GuardrailError, match="RGB"):
        validate_input(img)


def test_rejects_too_small():
    img = make_image(w=50, h=50)
    with pytest.raises(GuardrailError, match="too small"):
        validate_input(img)


def test_rejects_too_large():
    img = make_image(w=5000, h=5000)
    with pytest.raises(GuardrailError, match="too large"):
        validate_input(img)


def test_rejects_blank_image():
    img = make_image(value=0)  # all black
    with pytest.raises(GuardrailError, match="blank"):
        validate_input(img)


def test_rejects_overexposed_image():
    img = make_image(value=255)  # all white
    with pytest.raises(GuardrailError, match="overexposed"):
        validate_input(img)


# --- Output guardrails ---

def test_filters_low_confidence_detections():
    detections = [
        {"box": [10, 10, 50, 50], "score": 0.8},
        {"box": [60, 60, 90, 90], "score": 0.2},  # below threshold
    ]
    result = validate_output(detections, confidence_threshold=0.4)
    assert len(result["detections"]) == 1
    assert result["detections"][0]["score"] == 0.8


def test_warns_zero_detections():
    result = validate_output([], confidence_threshold=0.4)
    assert result["warnings"] == ["no_detections"]


def test_warns_too_many_detections():
    detections = [{"box": [i, i, i+10, i+10], "score": 0.9} for i in range(200)]
    result = validate_output(detections, confidence_threshold=0.4)
    assert "too_many_detections" in result["warnings"]


def test_clean_output_no_warnings():
    detections = [{"box": [10, 10, 50, 50], "score": 0.7}]
    result = validate_output(detections, confidence_threshold=0.4)
    assert result["warnings"] == []
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/model/test_guardrails.py -v
```

Expected: `ImportError`.

**Step 3: Implement `src/model/guardrails.py`**

```python
# src/model/guardrails.py
import numpy as np
from PIL import Image


class GuardrailError(ValueError):
    """Raised when an image fails input validation."""
    pass


MIN_DIM = 100
MAX_DIM = 4096
BLANK_THRESHOLD = 5       # mean pixel value below this → blank
OVEREXPOSED_THRESHOLD = 250  # mean pixel value above this → overexposed
MAX_DETECTIONS = 150


def validate_input(img: Image.Image) -> None:
    """
    Validate image before inference. Raises GuardrailError with a
    descriptive message if the image should not be processed.
    """
    if img.mode != "RGB":
        raise GuardrailError(f"Image must be RGB, got {img.mode}")

    w, h = img.size
    if min(w, h) < MIN_DIM:
        raise GuardrailError(f"Image too small: {w}x{h} (minimum {MIN_DIM}px on shortest edge)")
    if max(w, h) > MAX_DIM:
        raise GuardrailError(f"Image too large: {w}x{h} (maximum {MAX_DIM}px on longest edge)")

    mean_val = np.array(img).mean()
    if mean_val < BLANK_THRESHOLD:
        raise GuardrailError(f"Image appears blank (mean pixel value: {mean_val:.1f})")
    if mean_val > OVEREXPOSED_THRESHOLD:
        raise GuardrailError(f"Image appears overexposed (mean pixel value: {mean_val:.1f})")


def validate_output(
    detections: list[dict],
    confidence_threshold: float = 0.4,
) -> dict:
    """
    Filter detections by confidence and attach warnings for anomalous outputs.
    Returns dict with 'detections' (filtered) and 'warnings' (list of strings).
    """
    filtered = [d for d in detections if d["score"] >= confidence_threshold]
    warnings = []

    if len(filtered) == 0:
        warnings.append("no_detections")
    elif len(filtered) > MAX_DETECTIONS:
        warnings.append("too_many_detections")

    return {"detections": filtered, "warnings": warnings}
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/model/test_guardrails.py -v
```

Expected: all 10 tests PASS.

**Step 5: Commit**

```bash
git add src/model/guardrails.py tests/model/test_guardrails.py
git commit -m "feat: input/output guardrails for inference pipeline"
```

---

## Task 5: YOLOS-small detector wrapper (`src/model/detector.py`)

**Files:**
- Create: `src/model/detector.py`
- Create: `tests/model/test_detector.py`

**Step 1: Write the failing tests**

```python
# tests/model/test_detector.py
from unittest.mock import patch, MagicMock
import torch
import pytest
from PIL import Image
import numpy as np
from src.model.detector import SpaceDetector


@pytest.fixture
def mock_detector():
    """Detector with mocked HuggingFace model to avoid download in tests."""
    with patch("src.model.detector.YolosForObjectDetection.from_pretrained") as mock_model, \
         patch("src.model.detector.YolosImageProcessor.from_pretrained") as mock_proc:

        # Mock processor: returns a dict with input_ids tensor
        mock_proc.return_value.return_value = {
            "pixel_values": torch.zeros(1, 3, 512, 512)
        }
        mock_proc.return_value.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.85, 0.3]),
                "boxes": torch.tensor([[10., 10., 50., 50.], [60., 60., 80., 80.]]),
            }
        ]

        mock_model.return_value.return_value = MagicMock()
        mock_model.return_value.config.id2label = {0: "cat"}  # dummy COCO label

        detector = SpaceDetector()
        yield detector


def test_detector_initializes(mock_detector):
    assert mock_detector is not None


def test_detect_returns_list(mock_detector):
    img = Image.fromarray(np.random.randint(10, 200, (640, 640, 3), dtype=np.uint8))
    result = mock_detector.detect(img)
    assert isinstance(result, list)


def test_detect_result_has_expected_keys(mock_detector):
    img = Image.fromarray(np.random.randint(10, 200, (640, 640, 3), dtype=np.uint8))
    result = mock_detector.detect(img)
    for det in result:
        assert "box" in det
        assert "score" in det
        assert "label" not in det  # we strip class labels in base version
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/model/test_detector.py -v
```

Expected: `ImportError`.

**Step 3: Implement `src/model/detector.py`**

```python
# src/model/detector.py
import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

MODEL_NAME = "hustvl/yolos-small"


class SpaceDetector:
    """
    Wraps YOLOS-small for generic object detection on astronomical images.
    Returns bounding boxes + confidence scores only — no class labels.
    Class labels from COCO are intentionally stripped: in the base pipeline
    the model is used as a generic blob detector, not a classifier.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.processor = YolosImageProcessor.from_pretrained(model_name)
        self.model = YolosForObjectDetection.from_pretrained(model_name)
        self.model.eval()

    def detect(self, img: Image.Image) -> list[dict]:
        """
        Run inference on a PIL RGB image.
        Returns list of dicts: [{"box": [x1,y1,x2,y2], "score": float}, ...]
        """
        inputs = self.processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=0.0,  # return all; confidence filtering happens in guardrails
            target_sizes=target_sizes,
        )[0]

        detections = []
        for score, box in zip(results["scores"], results["boxes"]):
            detections.append({
                "box": [round(v, 2) for v in box.tolist()],
                "score": round(score.item(), 4),
            })

        return detections
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/model/test_detector.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add src/model/detector.py tests/model/test_detector.py
git commit -m "feat: YOLOS-small detector wrapper (bounding boxes + confidence only)"
```

---

## Task 6: MLflow pipeline orchestration (`src/pipeline.py`)

**Files:**
- Create: `src/pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
from src.pipeline import run_pipeline


@pytest.fixture
def pipeline_mocks(tmp_path):
    """Mock all external calls: SDSS, model download, MLflow."""
    from PIL import Image
    import numpy as np

    fake_img = Image.fromarray(np.random.randint(10, 200, (640, 640, 3), dtype=np.uint8))

    with patch("src.pipeline.build_dataset", return_value={"downloaded": 3, "skipped": 1}) as mock_ingest, \
         patch("src.pipeline.SpaceDetector") as mock_det_cls, \
         patch("src.pipeline.mlflow") as mock_mlflow, \
         patch("src.pipeline.Image.open", return_value=fake_img):

        mock_det = MagicMock()
        mock_det.detect.return_value = [{"box": [10, 10, 50, 50], "score": 0.75}]
        mock_det_cls.return_value = mock_det

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        yield {
            "tmp_path": tmp_path,
            "mock_ingest": mock_ingest,
            "mock_mlflow": mock_mlflow,
        }


def test_pipeline_runs_without_error(pipeline_mocks):
    run_pipeline(
        data_dir=pipeline_mocks["tmp_path"],
        n_regions=3,
        confidence_threshold=0.4,
    )


def test_pipeline_calls_mlflow_log_params(pipeline_mocks):
    run_pipeline(
        data_dir=pipeline_mocks["tmp_path"],
        n_regions=3,
        confidence_threshold=0.4,
    )
    pipeline_mocks["mock_mlflow"].log_params.assert_called_once()


def test_pipeline_calls_mlflow_log_metrics(pipeline_mocks):
    run_pipeline(
        data_dir=pipeline_mocks["tmp_path"],
        n_regions=3,
        confidence_threshold=0.4,
    )
    pipeline_mocks["mock_mlflow"].log_metrics.assert_called_once()
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: `ImportError`.

**Step 3: Implement `src/pipeline.py`**

```python
# src/pipeline.py
import json
import mlflow
import mlflow.pytorch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from src.data.ingest import build_dataset
from src.data.preprocess import preprocess_image
from src.model.detector import SpaceDetector
from src.model.guardrails import validate_input, validate_output, GuardrailError

EXPERIMENT_NAME = "space-object-detection"


def _draw_detections(img: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw bounding boxes on image for artifact logging."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), f"{det['score']:.2f}", fill="red")
    return out


def run_pipeline(
    data_dir: Path = Path("data"),
    n_regions: int = 20,
    radius_deg: float = 0.05,
    scale: float = 0.2,
    confidence_threshold: float = 0.4,
):
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # --- Log params ---
        mlflow.log_params({
            "n_regions": n_regions,
            "radius_deg": radius_deg,
            "scale": scale,
            "confidence_threshold": confidence_threshold,
            "model_name": "hustvl/yolos-small",
        })

        # --- Ingest ---
        print(f"[1/4] Ingesting data: n_regions={n_regions}")
        stats = build_dataset(data_dir, n_regions=n_regions, radius_deg=radius_deg, scale=scale)
        mlflow.log_metrics({
            "n_images_downloaded": stats["downloaded"],
            "n_regions_skipped": stats["skipped"],
        })

        # --- Load model ---
        print("[2/4] Loading YOLOS-small")
        detector = SpaceDetector()

        # --- Inference over all downloaded images ---
        print("[3/4] Running inference")
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        all_detections = []
        guardrail_rejections = 0
        all_scores = []

        image_paths = sorted(raw_dir.glob("*.jpg"))

        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")

            # Input guardrail
            try:
                validate_input(img)
            except GuardrailError as e:
                print(f"  [guardrail] Rejected {img_path.name}: {e}")
                guardrail_rejections += 1
                continue

            # Preprocess + detect
            preprocessed = preprocess_image(img)
            raw_detections = detector.detect(preprocessed)

            # Output guardrail
            result = validate_output(raw_detections, confidence_threshold=confidence_threshold)
            detections = result["detections"]

            if result["warnings"]:
                print(f"  [guardrail] {img_path.name}: {result['warnings']}")

            all_detections.append({"image": img_path.name, "detections": detections})
            all_scores.extend([d["score"] for d in detections])

            # Save annotated image as artifact
            annotated = _draw_detections(img, detections)
            annotated.save(processed_dir / img_path.name)

        # --- Log metrics ---
        total_detections = sum(len(r["detections"]) for r in all_detections)
        avg_confidence = float(np.mean(all_scores)) if all_scores else 0.0

        mlflow.log_metrics({
            "n_detections_total": total_detections,
            "avg_confidence": avg_confidence,
            "guardrail_rejections": guardrail_rejections,
        })

        # --- Log artifacts ---
        print("[4/4] Logging artifacts")
        detections_path = data_dir / "detections.json"
        detections_path.write_text(json.dumps(all_detections, indent=2))
        mlflow.log_artifact(str(detections_path))

        for img_path in list((processed_dir).glob("*.jpg"))[:5]:  # log up to 5 sample images
            mlflow.log_artifact(str(img_path), artifact_path="annotated_samples")

        # --- Register model ---
        mlflow.pytorch.log_model(
            detector.model,
            artifact_path="model",
            registered_model_name="space-detector",
        )

        print(f"Done. {total_detections} detections across {len(image_paths)} images.")
        print(f"Guardrail rejections: {guardrail_rejections}")
        print(f"Average confidence: {avg_confidence:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run space object detection pipeline")
    parser.add_argument("--n-regions", type=int, default=20)
    parser.add_argument("--radius-deg", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        n_regions=args.n_regions,
        radius_deg=args.radius_deg,
        scale=args.scale,
        confidence_threshold=args.confidence_threshold,
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: MLflow pipeline orchestration with params, metrics, artifacts, model registry"
```

---

## Task 7: Inference script (`src/inference.py`)

**Files:**
- Create: `src/inference.py`

> No unit tests for this module — it's a thin CLI wrapper around the registered model and guardrails. Integration testing is done manually via `mlflow models serve`.

**Step 1: Implement `src/inference.py`**

```python
# src/inference.py
"""
Load the registered YOLOS-small model from MLflow registry and run inference.

Usage:
    # Start the MLflow model server first:
    mlflow models serve -m "models:/space-detector/1" --port 5001 --no-conda

    # Then call the endpoint:
    python src/inference.py --image data/raw/field_0000.jpg
"""
import json
import argparse
import requests
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO

from src.model.guardrails import validate_input, GuardrailError
from src.data.preprocess import preprocess_image


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_inference(image_path: Path, endpoint: str = "http://localhost:5001") -> dict:
    img = Image.open(image_path).convert("RGB")

    # Guardrail: validate before sending
    validate_input(img)

    preprocessed = preprocess_image(img)
    payload = {"instances": [{"b64": image_to_base64(preprocessed)}]}

    response = requests.post(
        f"{endpoint}/invocations",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a space image")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--endpoint", default="http://localhost:5001")
    args = parser.parse_args()

    try:
        result = run_inference(args.image, args.endpoint)
        print(json.dumps(result, indent=2))
    except GuardrailError as e:
        print(f"Guardrail rejected image: {e}")
```

**Step 2: Commit**

```bash
git add src/inference.py
git commit -m "feat: inference script for mlflow models serve endpoint"
```

---

## Task 8: Run full test suite and verify

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

**Step 2: Run pipeline end-to-end (small)**

```bash
uv run python src/pipeline.py --n-regions 3 --data-dir data
```

Expected: downloads 3 images, runs inference, logs to MLflow.

**Step 3: Inspect MLflow UI**

```bash
uv run mlflow ui --port 5000
```

Open `http://localhost:5000` — verify:
- Experiment `space-object-detection` exists
- Run shows params: `n_regions`, `radius_deg`, `scale`, `confidence_threshold`, `model_name`
- Run shows metrics: `n_images_downloaded`, `n_detections_total`, `avg_confidence`, `guardrail_rejections`
- Artifacts: `detections.json`, `annotated_samples/`
- Model registered as `space-detector` in Model Registry

**Step 4: Serve model and test endpoint**

```bash
# Terminal 1
uv run mlflow models serve -m "models:/space-detector/1" --port 5001 --no-conda

# Terminal 2
uv run python src/inference.py --image data/raw/field_0000.jpg
```

**Step 5: Final commit**

```bash
git add .
git commit -m "chore: verify full pipeline end-to-end"
```

---

## Task 9: README and requirements.txt

**Files:**
- Modify: `README.md`
- Create: `requirements.txt` (exported from uv for reference)

**Step 1: Export requirements**

```bash
uv pip freeze > requirements.txt
```

**Step 2: Write README.md**

```markdown
# Space Object Detection — ML Pipeline

End-to-end ML pipeline for detecting objects in astronomical sky survey images
using YOLOS-small (Vision Transformer) and MLflow for experiment tracking.

## Setup

```bash
uv sync
```

## Run the pipeline

```bash
uv run python src/pipeline.py --n-regions 20 --confidence-threshold 0.4
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
uv run python src/inference.py --image data/raw/field_0000.jpg
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
```

**Step 3: Commit**

```bash
git add README.md requirements.txt
git commit -m "docs: README with setup and usage instructions"
```

---

## Execution options

**Option 1 — Subagent-Driven (this session)**
Dispatch a fresh subagent per task using the `subagent-driven-development` skill. Review output between tasks.

**Option 2 — Parallel Session (separate)**
Open a new Claude Code session, load this plan with `/executing-plans`, and run tasks in batches with checkpoints.
