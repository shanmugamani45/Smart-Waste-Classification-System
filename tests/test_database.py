import pytest
import tempfile
import os
from database.database import DatabaseManager

@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    db = DatabaseManager(db_path=db_path)
    yield db
    db.close()
    import gc
    gc.collect()  # Release any lingering SQLite handles on Windows
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except Exception:
        pass

def test_insert_and_get_history(temp_db):
    pred_id = temp_db.insert_prediction(
        predicted_class="plastic",
        confidence=0.965,
        recommended_action="Recycle",
        circular_strategy="Reuse → Recycle",
        environmental_priority="HIGH",
        image_path="/test/plastic1.jpg"
    )
    assert pred_id > 0

    history = temp_db.get_history(limit=10)
    assert len(history) == 1
    assert history[0]["predicted_class"] == "plastic"
    assert history[0]["confidence"] == 0.965
    assert history[0]["recommended_action"] == "Recycle"

def test_feedback_update(temp_db):
    pred_id = temp_db.insert_prediction(
        predicted_class="metal",
        confidence=0.72,
        recommended_action="Recycle",
        circular_strategy="Recycle",
        environmental_priority="HIGH"
    )

    success = temp_db.update_feedback(pred_id, user_feedback="incorrect", corrected_class="plastic")
    assert success is True

    record = temp_db.get_prediction_by_id(pred_id)
    assert record["user_feedback"] == "incorrect"
    assert record["corrected_class"] == "plastic"

def test_statistics_aggregation(temp_db):
    # Insert multiple items
    temp_db.insert_prediction("plastic", 0.90, "Recycle", "Reuse → Recycle", "HIGH")
    temp_db.insert_prediction("plastic", 0.85, "Recycle", "Reuse → Recycle", "HIGH")
    temp_db.insert_prediction("e_waste", 0.95, "Specialized E-Waste Processing", "Repair → Refurbish", "HIGH")

    stats = temp_db.get_statistics()
    assert stats["total_predictions"] == 3
    assert stats["category_distribution"]["plastic"] == 2
    assert stats["category_distribution"]["e_waste"] == 1
    assert stats["category_percentages"]["plastic"] == pytest.approx(66.67, 0.01)
    assert stats["action_distribution"]["Recycle"] == 2
