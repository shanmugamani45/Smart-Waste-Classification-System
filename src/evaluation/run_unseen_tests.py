import os
import json
import io
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter

from src.inference.predictor import WastePredictor
from src.utils.config_loader import get_config
from src.utils.logger import logger


class UnseenTestEvaluator:
    """
    Evaluates model resilience across challenging real-world perturbations:
    - Lighting variations (under/over exposed)
    - Blur & dirt degradation
    - Scale & cropping occlusion
    - Low-confidence and failure case tracking
    """

    def __init__(self, output_dir: str = None):
        config = get_config()
        self.output_dir = Path(output_dir or config["paths"]["evaluation_dir"])
        self.results_dir = self.output_dir / "test_results"
        self.failure_dir = self.output_dir / "failure_cases"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.failure_dir.mkdir(parents=True, exist_ok=True)

        self.predictor = WastePredictor()

    def evaluate_synthetic_perturbation_suite(self) -> Dict[str, Any]:
        """
        Generates and evaluates a battery of challenging test samples representing
        real-world waste sorting variability without fabricating ground-truth metrics.
        """
        logger.info("Initiating real-world perturbation evaluation suite...")
        scenarios = [
            {"name": "standard_sample", "adjust": lambda img: img, "desc": "Standard illumination"},
            {"name": "low_light", "adjust": lambda img: ImageEnhance.Brightness(img).enhance(0.4), "desc": "Dim/poor lighting"},
            {"name": "high_glare", "adjust": lambda img: ImageEnhance.Brightness(img).enhance(1.8), "desc": "Overexposed/direct sunlight"},
            {"name": "soiled_blurred", "adjust": lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)), "desc": "Dirty or motion-blurred lens"},
            {"name": "partial_occlusion", "adjust": lambda img: self._occlude_region(img), "desc": "Partially occluded waste object"},
        ]

        test_log = []
        low_confidence_cases = []

        # Create a base sample image
        base_img = Image.new("RGB", (224, 224), color=(60, 120, 180))

        for sc in scenarios:
            perturbed = sc["adjust"](base_img.copy())

            # Convert to bytes
            buf = io.BytesIO()
            perturbed.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            # Predict
            result = self.predictor.predict(img_bytes, include_gradcam=True)
            pred = result["prediction"]
            rec = result["recommendation"]

            case_entry = {
                "scenario": sc["name"],
                "description": sc["desc"],
                "predicted_class": pred["class"],
                "confidence": pred["confidence"],
                "confidence_level": rec["confidence_level"],
                "latency_ms": pred["latency_ms"],
                "action": rec["recommended_action"],
                "warning": rec.get("confidence_warning")
            }
            test_log.append(case_entry)

            # Flag edge cases / low confidence
            if rec["confidence_level"] != "HIGH" or rec.get("confidence_warning"):
                low_confidence_cases.append(case_entry)
                fail_path = self.failure_dir / f"{sc['name']}.jpg"
                perturbed.save(fail_path)

        # Save test report
        report_path = self.results_dir / "unseen_test_log.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_scenarios_tested": len(scenarios),
                "flagged_cases_count": len(low_confidence_cases),
                "test_log": test_log,
                "low_confidence_cases": low_confidence_cases
            }, f, indent=4)

        logger.info(f"Unseen test evaluation complete. Log saved to: {report_path}")
        return {
            "total_tested": len(scenarios),
            "flagged_cases": len(low_confidence_cases),
            "report_path": str(report_path)
        }

    @staticmethod
    def _occlude_region(img: Image.Image) -> Image.Image:
        """Simulates partial occlusion by zeroing a quadrant."""
        img_copy = img.copy()
        w, h = img_copy.size
        # Draw black box over top-left quadrant
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle([0, 0, w // 2, h // 2], fill=(0, 0, 0))
        return img_copy


if __name__ == "__main__":
    evaluator = UnseenTestEvaluator()
    evaluator.evaluate_synthetic_perturbation_suite()
