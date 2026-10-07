import os
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.utils.config_loader import get_config
from src.utils.logger import logger
from database.models import CREATE_PREDICTIONS_TABLE, CREATE_FEEDBACK_INDEX, CREATE_CLASS_INDEX


class DatabaseManager:
    """Thread-safe SQLite connection and analytics query manager."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = str(Path(db_path).resolve())
        else:
            config = get_config()
            self.db_path = config["paths"]["database"]

        self._ensure_dir()
        self.init_db()

    def _ensure_dir(self) -> None:
        """Ensures the directory hosting the SQLite database exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a new SQLite connection with dict-like row access."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self) -> None:
        """Closes all cached resources if needed."""
        pass

    def init_db(self) -> None:
        """Applies schema DDL to initialize database tables."""
        from database.models import CREATE_FEEDBACK_CACHE_TABLE
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(CREATE_PREDICTIONS_TABLE)
            cursor.execute(CREATE_FEEDBACK_INDEX)
            cursor.execute(CREATE_CLASS_INDEX)
            cursor.execute(CREATE_FEEDBACK_CACHE_TABLE)
            conn.commit()
        logger.info(f"Database initialized at: {self.db_path}")

    def cache_image_feedback(self, image_hash: str, corrected_class: str) -> None:
        """Caches verified human classification for an exact image hash."""
        sql = """
        INSERT OR REPLACE INTO image_feedback_cache (image_hash, corrected_class, timestamp)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (image_hash, corrected_class))
            conn.commit()

    def get_cached_feedback_by_hash(self, image_hash: str) -> Optional[str]:
        """Looks up if this exact image was previously corrected or verified by user feedback."""
        sql = "SELECT corrected_class FROM image_feedback_cache WHERE image_hash = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (image_hash,))
            row = cursor.fetchone()
            return row["corrected_class"] if row else None

    def get_all_cached_feedback(self) -> Dict[str, str]:
        """Returns all cached (image_hash -> corrected_class) mappings."""
        sql = "SELECT image_hash, corrected_class FROM image_feedback_cache"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return {row["image_hash"]: row["corrected_class"] for row in cursor.fetchall()}

    def insert_prediction(
        self,
        predicted_class: str,
        confidence: float,
        recommended_action: str,
        circular_strategy: str,
        environmental_priority: str,
        image_path: Optional[str] = None
    ) -> int:
        """Inserts a new prediction log and returns the inserted record ID."""
        sql = """
        INSERT INTO predictions (
            predicted_class, confidence, recommended_action,
            circular_strategy, environmental_priority, image_path
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                sql,
                (
                    predicted_class,
                    float(confidence),
                    recommended_action,
                    circular_strategy,
                    environmental_priority,
                    image_path
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieves paginated historical predictions ordered by most recent."""
        sql = """
        SELECT id, timestamp, predicted_class, confidence,
               recommended_action, circular_strategy, environmental_priority,
               image_path, user_feedback, corrected_class
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (limit, offset))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a single prediction by ID."""
        sql = "SELECT * FROM predictions WHERE id = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (prediction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_feedback(
        self,
        prediction_id: int,
        user_feedback: str,
        corrected_class: Optional[str] = None
    ) -> bool:
        """Updates human-in-the-loop audit feedback for a prediction."""
        sql = """
        UPDATE predictions
        SET user_feedback = ?, corrected_class = ?
        WHERE id = ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (user_feedback, corrected_class, prediction_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates live analytics distributions from the predictions table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Total count & average confidence
            cursor.execute("SELECT COUNT(*), AVG(confidence) FROM predictions")
            row = cursor.fetchone()
            total_predictions = row[0] or 0
            avg_confidence = round(row[1] or 0.0, 4)

            # 2. Waste Category Distribution
            cursor.execute("""
                SELECT predicted_class, COUNT(*) as count
                FROM predictions
                GROUP BY predicted_class
                ORDER BY count DESC
            """)
            category_counts = {r["predicted_class"]: r["count"] for r in cursor.fetchall()}

            # 3. Circular Action Distribution
            cursor.execute("""
                SELECT recommended_action, COUNT(*) as count
                FROM predictions
                GROUP BY recommended_action
                ORDER BY count DESC
            """)
            action_counts = {r["recommended_action"]: r["count"] for r in cursor.fetchall()}

            # 4. Environmental Priority Distribution
            cursor.execute("""
                SELECT environmental_priority, COUNT(*) as count
                FROM predictions
                GROUP BY environmental_priority
            """)
            priority_counts = {r["environmental_priority"]: r["count"] for r in cursor.fetchall()}

            # Compute percentages for categories
            category_percentages = {}
            if total_predictions > 0:
                for cat, count in category_counts.items():
                    category_percentages[cat] = round((count / total_predictions) * 100, 2)

            return {
                "total_predictions": total_predictions,
                "average_confidence": avg_confidence,
                "category_distribution": category_counts,
                "category_percentages": category_percentages,
                "action_distribution": action_counts,
                "priority_distribution": priority_counts
            }
