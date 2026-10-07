"""Database table definitions and DDL schemas for the Circular Economy system."""

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    recommended_action TEXT NOT NULL,
    circular_strategy TEXT NOT NULL,
    environmental_priority TEXT NOT NULL,
    image_path TEXT,
    user_feedback TEXT DEFAULT NULL,
    corrected_class TEXT DEFAULT NULL
);
"""

CREATE_FEEDBACK_INDEX = """
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (timestamp DESC);
"""

CREATE_CLASS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions (predicted_class);
"""

CREATE_FEEDBACK_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS image_feedback_cache (
    image_hash TEXT PRIMARY KEY,
    corrected_class TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""
