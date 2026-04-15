# src/model/detector.py
from pathlib import Path

import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

MODEL_NAME = "hustvl/yolos-small"

# COCO-pretrained model has no meaningful astronomical labels — strip them.
# A fine-tuned model exposes id2label = {0: "star", 1: "galaxy", 2: "quasar"},
# which is included in the detection output automatically.
_COCO_MODEL_NAME = "hustvl/yolos-small"


class SpaceDetector:
    """
    Wraps YOLOS-small for object detection on astronomical images.

    When loaded from the base HuggingFace checkpoint (COCO-pretrained) class
    labels are stripped — the model is used as a generic blob detector.
    When loaded from a fine-tuned local checkpoint the astronomical class label
    (star / galaxy / quasar) is included in each detection dict.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = str(model_name)
        self.processor = YolosImageProcessor.from_pretrained(model_name)
        self.model = YolosForObjectDetection.from_pretrained(model_name)
        self.model.eval()

        # Determine whether this checkpoint has astronomical labels.
        # A fine-tuned model will have id2label = {0: "star", ...}.
        id2label = getattr(self.model.config, "id2label", {})
        self._include_labels = bool(id2label) and 0 in id2label and id2label[0] in ("star", "galaxy", "quasar")

    @classmethod
    def from_finetuned(cls, checkpoint_dir: str | Path) -> "SpaceDetector":
        """Load from a local fine-tuned checkpoint directory."""
        return cls(model_name=str(checkpoint_dir))

    def detect(self, img: Image.Image) -> list[dict]:
        """
        Run inference on a PIL RGB image.

        Returns list of dicts:
          - Base model:       [{"box": [x1,y1,x2,y2], "score": float}, ...]
          - Fine-tuned model: [{"box": [...], "score": float, "label": str}, ...]
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

        id2label = self.model.config.id2label
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            det: dict = {
                "box": [round(v, 2) for v in box.tolist()],
                "score": round(score.item(), 4),
            }
            if self._include_labels:
                det["label"] = id2label[label.item()]
            detections.append(det)

        return detections
