from typing import Dict, Any, Optional
from src.utils.config_loader import get_config
from src.utils.logger import logger


class CircularRecommendationEngine:
    """
    Decoupled rule-based policy engine mapping classified waste materials
    to circular economy strategies, actions, and environmental priorities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.rules = self.config.get("circular_rules", {})
        self.classes = self.config.get("classes", [])

        # Confidence thresholds
        inference_cfg = self.config.get("inference", {})
        self.high_threshold = inference_cfg.get("confidence_high_threshold", 0.80)
        self.moderate_threshold = inference_cfg.get("confidence_moderate_threshold", 0.50)

    def evaluate(self, predicted_class: str, confidence: float) -> Dict[str, Any]:
        """
        Evaluates a classified waste item and outputs its circular economy directive.

        Args:
            predicted_class: Normalized class label (e.g. 'plastic', 'e_waste')
            confidence: Probability score in [0.0, 1.0]

        Returns:
            Dictionary containing circular action, strategy, priority, confidence level,
            and action guidelines.
        """
        norm_class = predicted_class.strip().lower()

        if norm_class not in self.rules:
            logger.warning(f"Unknown class '{predicted_class}' submitted to recommendation engine.")
            return {
                "class": norm_class,
                "confidence": round(confidence, 4),
                "confidence_level": self._classify_confidence(confidence),
                "confidence_warning": self._get_confidence_warning(confidence),
                "recommended_action": "Manual Sorting Required",
                "circular_strategy": "Direct to local municipal sorting facility",
                "environmental_priority": "MEDIUM",
                "guidelines": "Unrecognized waste category. Please verify material type manually before disposal."
            }

        rule = self.rules[norm_class]
        confidence_level = self._classify_confidence(confidence)
        confidence_warning = self._get_confidence_warning(confidence)

        return {
            "class": norm_class,
            "confidence": round(float(confidence), 4),
            "confidence_level": confidence_level,
            "confidence_warning": confidence_warning,
            "recommended_action": rule.get("recommended_action", "Dispose Responsibly"),
            "circular_strategy": rule.get("circular_strategy", "Assess material recyclability"),
            "environmental_priority": rule.get("environmental_priority", "MEDIUM"),
            "guidelines": rule.get("guidelines", "Follow local municipal waste disposal guidance.")
        }

    def _classify_confidence(self, confidence: float) -> str:
        """Classifies confidence into qualitative evaluation tiers."""
        if confidence >= self.high_threshold:
            return "HIGH"
        elif confidence >= self.moderate_threshold:
            return "MODERATE"
        else:
            return "LOW"

    def _get_confidence_warning(self, confidence: float) -> Optional[str]:
        """Generates warning strings if confidence fails high threshold."""
        if confidence < self.moderate_threshold:
            return "Low confidence - manual verification recommended before circular routing."
        elif confidence < self.high_threshold:
            return "Moderate confidence - inspect item details to ensure proper material sorting."
        return None
