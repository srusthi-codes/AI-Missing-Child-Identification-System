from config.constants import MAX_CHILD_AGE
from database.connection import get_connection
from utils.logger import get_logger


logger = get_logger(__name__)


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('parent_guardian', 'child_finder', 'authority')),
        phone TEXT,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_login_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS missing_children (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT NOT NULL UNIQUE,
        full_name TEXT NOT NULL,
        age INTEGER NOT NULL CHECK(age >= 0 AND age <= 100),
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
    """
    CREATE TABLE IF NOT EXISTS found_child_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reporter_user_id INTEGER NOT NULL,
        search_id INTEGER,
        uploaded_image_path TEXT NOT NULL,
        location TEXT NOT NULL,
        description TEXT,
        matches_found INTEGER NOT NULL DEFAULT 0,
        best_similarity_score REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending_verification',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (reporter_user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (search_id) REFERENCES search_history(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS found_child_report_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_id INTEGER NOT NULL,
        child_id INTEGER NOT NULL,
        case_id TEXT NOT NULL,
        similarity_score REAL NOT NULL,
        match_rank INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (report_id) REFERENCES found_child_reports(id) ON DELETE CASCADE,
        FOREIGN KEY (child_id) REFERENCES missing_children(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipient_user_id INTEGER,
        recipient_role TEXT,
        child_id INTEGER,
        report_id INTEGER,
        case_id TEXT,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        notification_type TEXT NOT NULL,
        is_read INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recipient_user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (child_id) REFERENCES missing_children(id) ON DELETE CASCADE,
        FOREIGN KEY (report_id) REFERENCES found_child_reports(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS age_progression_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        child_id INTEGER NOT NULL,
        case_id TEXT NOT NULL,
        source_image_path TEXT NOT NULL,
        generated_image_path TEXT NOT NULL,
        source_age INTEGER NOT NULL,
        target_age INTEGER NOT NULL,
        target_age_label TEXT NOT NULL,
        progression_years INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        identity_score REAL,
        identity_quality TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (child_id) REFERENCES missing_children(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
    "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)",
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
    "CREATE INDEX IF NOT EXISTS idx_found_reports_reporter ON found_child_reports(reporter_user_id)",
    "CREATE INDEX IF NOT EXISTS idx_found_reports_status ON found_child_reports(status)",
    "CREATE INDEX IF NOT EXISTS idx_found_reports_created_at ON found_child_reports(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_found_report_matches_report ON found_child_report_matches(report_id)",
    "CREATE INDEX IF NOT EXISTS idx_found_report_matches_child ON found_child_report_matches(child_id)",
    "CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(recipient_user_id)",
    "CREATE INDEX IF NOT EXISTS idx_notifications_role ON notifications(recipient_role)",
    "CREATE INDEX IF NOT EXISTS idx_notifications_child ON notifications(child_id)",
    "CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_age_progression_child_id ON age_progression_history(child_id)",
    "CREATE INDEX IF NOT EXISTS idx_age_progression_case_id ON age_progression_history(case_id)",
    "CREATE INDEX IF NOT EXISTS idx_age_progression_created_at ON age_progression_history(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_activity_logs_created_at ON activity_logs(created_at)",
]


def initialize_database() -> None:
    connection = get_connection()
    try:
        _migrate_missing_children_age_range(connection)
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


def _migrate_missing_children_age_range(connection) -> None:
    row = connection.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'missing_children'
        """
    ).fetchone()
    if row is None:
        return

    table_sql = str(row["sql"] or "")
    normalized_sql = table_sql.replace(" ", "").lower()
    if "age>=0andage<=18" not in normalized_sql:
        return

    logger.info("Migrating missing_children age CHECK constraint to 0-%s", MAX_CHILD_AGE)
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("PRAGMA legacy_alter_table = ON")
    connection.execute("ALTER TABLE missing_children RENAME TO missing_children_legacy_age_check")
    connection.execute(
        """
        CREATE TABLE missing_children (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL UNIQUE,
            full_name TEXT NOT NULL,
            age INTEGER NOT NULL CHECK(age >= 0 AND age <= 100),
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
        """
    )
    connection.execute(
        """
        INSERT INTO missing_children (
            id,
            case_id,
            full_name,
            age,
            gender,
            identification_marks,
            last_seen_location,
            last_seen_date,
            last_seen_time,
            description,
            status,
            registered_by,
            created_at,
            updated_at
        )
        SELECT
            id,
            case_id,
            full_name,
            age,
            gender,
            identification_marks,
            last_seen_location,
            last_seen_date,
            last_seen_time,
            description,
            status,
            registered_by,
            created_at,
            updated_at
        FROM missing_children_legacy_age_check
        """
    )
    connection.execute("DROP TABLE missing_children_legacy_age_check")
    connection.execute("PRAGMA legacy_alter_table = OFF")
    connection.execute("PRAGMA foreign_keys = ON")
