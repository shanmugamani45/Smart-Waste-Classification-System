import pytest
from src.recommendation.recommendation_engine import CircularRecommendationEngine

@pytest.fixture
def engine():
    return CircularRecommendationEngine()

def test_plastic_recommendation(engine):
    res = engine.evaluate("plastic", 0.95)
    assert res["class"] == "plastic"
    assert res["confidence"] == 0.95
    assert res["confidence_level"] == "HIGH"
    assert res["confidence_warning"] is None
    assert res["recommended_action"] == "Recycle"
    assert "Reuse → Recycle" in res["circular_strategy"]
    assert res["environmental_priority"] == "HIGH"

def test_e_waste_recommendation(engine):
    res = engine.evaluate("e_waste", 0.88)
    assert res["class"] == "e_waste"
    assert "Repair → Refurbish" in res["circular_strategy"]
    assert res["environmental_priority"] == "HIGH"

def test_moderate_confidence_warning(engine):
    res = engine.evaluate("metal", 0.65)
    assert res["confidence_level"] == "MODERATE"
    assert res["confidence_warning"] is not None
    assert "Moderate confidence" in res["confidence_warning"]

def test_low_confidence_warning(engine):
    res = engine.evaluate("glass", 0.35)
    assert res["confidence_level"] == "LOW"
    assert "Low confidence" in res["confidence_warning"]

def test_unrecognized_class_fallback(engine):
    res = engine.evaluate("hazardous_chemical", 0.90)
    assert res["recommended_action"] == "Manual Sorting Required"
    assert "Unrecognized waste category" in res["guidelines"]
