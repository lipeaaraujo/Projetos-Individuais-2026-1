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
