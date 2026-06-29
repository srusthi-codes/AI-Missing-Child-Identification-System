from database.connection import get_connection
from utils.logger import get_logger


logger = get_logger(__name__)


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS missing_children (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT NOT NULL UNIQUE,
        full_name TEXT NOT NULL,
        age INTEGER NOT NULL CHECK(age >= 0 AND age <= 18),
        gender TEXT NOT NULL,
        identification_marks TEXT,
        last_seen_location TEXT NOT NULL,
        last_seen_date TEXT NOT NULL,
        last_seen_time TEXT,
        description TEXT,
        status TEXT NOT NULL DEFAULT 'missing',
        registered_by INTEGER,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS parent_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        child_id INTEGER NOT NULL,
        guardian_name TEXT NOT NULL,
        relationship TEXT NOT NULL,
        phone TEXT NOT NULL,
        email TEXT,
        address TEXT NOT NULL,
        government_id_type TEXT,
        government_id_last4 TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (child_id) REFERENCES missing_children(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS child_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        child_id INTEGER NOT NULL,
        case_id TEXT NOT NULL,
        image_path TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        content_type TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        image_hash TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (child_id) REFERENCES missing_children(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        child_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        image_hash TEXT NOT NULL UNIQUE,
        embedding TEXT NOT NULL,
        model_name TEXT NOT NULL,
        detector_backend TEXT NOT NULL,
        embedding_dimension INTEGER NOT NULL,
        quality_score REAL NOT NULL,
        face_confidence REAL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (child_id) REFERENCES missing_children(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT NOT NULL,
        entity_type TEXT,
        entity_id INTEGER,
        ip_address TEXT,
        details TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS search_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uploaded_image_path TEXT NOT NULL,
        image_hash TEXT,
        matches_found INTEGER NOT NULL DEFAULT 0,
        best_similarity_score REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'completed',
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_missing_children_case_id ON missing_children(case_id)",
    "CREATE INDEX IF NOT EXISTS idx_missing_children_status ON missing_children(status)",
    "CREATE INDEX IF NOT EXISTS idx_missing_children_full_name ON missing_children(full_name)",
    "CREATE INDEX IF NOT EXISTS idx_parent_details_child_id ON parent_details(child_id)",
    "CREATE INDEX IF NOT EXISTS idx_parent_details_phone ON parent_details(phone)",
    "CREATE INDEX IF NOT EXISTS idx_child_images_child_id ON child_images(child_id)",
    "CREATE INDEX IF NOT EXISTS idx_child_images_hash ON child_images(image_hash)",
    "CREATE INDEX IF NOT EXISTS idx_face_embeddings_child_id ON face_embeddings(child_id)",
    "CREATE INDEX IF NOT EXISTS idx_face_embeddings_image_hash ON face_embeddings(image_hash)",
    "CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_search_history_image_hash ON search_history(image_hash)",
    "CREATE INDEX IF NOT EXISTS idx_activity_logs_created_at ON activity_logs(created_at)",
]


def initialize_database() -> None:
    connection = get_connection()
    try:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.commit()
        logger.info("Database schema initialized successfully")
    except Exception:
        connection.rollback()
        logger.exception("Database schema initialization failed")
        raise
    finally:
        connection.close()
