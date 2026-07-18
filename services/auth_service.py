import base64
import hashlib
import hmac
import secrets
import sqlite3
from typing import Any

from config.constants import ROLE_AUTHORITY, ROLE_FINDER, ROLE_LABELS, ROLE_PARENT, USER_ROLES
from database.connection import database_transaction
from database.repositories.log_repository import create_activity_log
from database.repositories.user_repository import create_user, fetch_user_by_email, fetch_user_by_id, update_user_last_login
from database.schema import initialize_database
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)

PASSWORD_ITERATIONS = 180_000
PASSWORD_SCHEME = "pbkdf2_sha256"

DEFAULT_USERS = (
    {
        "email": "parent@childshield.local",
        "password": "Parent@123",
        "full_name": "Demo Parent Guardian",
        "role": ROLE_PARENT,
        "phone": "+10000000001",
    },
    {
        "email": "finder@childshield.local",
        "password": "Finder@123",
        "full_name": "Demo Child Finder",
        "role": ROLE_FINDER,
        "phone": "+10000000002",
    },
    {
        "email": "authority@childshield.local",
        "password": "Authority@123",
        "full_name": "Demo Authority Officer",
        "role": ROLE_AUTHORITY,
        "phone": "+10000000003",
    },
)


class AuthenticationError(Exception):
    """Raised when authentication cannot be completed."""


class AuthorizationError(Exception):
    """Raised when a user attempts to access a role they do not own."""


def ensure_default_users() -> None:
    try:
        initialize_database()
        with database_transaction() as connection:
            for user in DEFAULT_USERS:
                if fetch_user_by_email(connection, user["email"]) is not None:
                    continue
                create_user(
                    connection,
                    {
                        "email": user["email"],
                        "password_hash": hash_password(user["password"]),
                        "full_name": user["full_name"],
                        "role": user["role"],
                        "phone": user["phone"],
                    },
                )
            create_activity_log(
                connection=connection,
                action="ensure_default_role_users",
                entity_type="users",
                details={"roles": [user["role"] for user in DEFAULT_USERS]},
            )
        logger.info("Default role users verified")
    except sqlite3.Error as exc:
        logger.exception("Database error while ensuring default users")
        raise AuthenticationError("Unable to prepare role-based login users.") from exc


def authenticate_user(email: str, password: str, expected_role: str) -> dict[str, Any]:
    normalized_email = _validate_email(email)
    if expected_role not in USER_ROLES:
        raise ValidationError("Selected login role is invalid.")
    if not password:
        raise ValidationError("Password is required.")

    try:
        initialize_database()
        with database_transaction() as connection:
            user = fetch_user_by_email(connection, normalized_email)
            if user is None or not user["is_active"]:
                raise AuthenticationError("Invalid email, password, or role.")
            if user["role"] != expected_role:
                raise AuthorizationError(f"Use the {ROLE_LABELS[user['role']]} login for this account.")
            if not verify_password(password, user["password_hash"]):
                raise AuthenticationError("Invalid email, password, or role.")

            update_user_last_login(connection, int(user["user_id"]))
            create_activity_log(
                connection=connection,
                action="user_login",
                entity_type="users",
                entity_id=int(user["user_id"]),
                user_id=int(user["user_id"]),
                details={"role": user["role"]},
            )

        logger.info("User authenticated user_id=%s role=%s", user["user_id"], user["role"])
        return public_user(user)
    except (AuthenticationError, AuthorizationError, ValidationError):
        raise
    except sqlite3.Error as exc:
        logger.exception("Database error during login")
        raise AuthenticationError("Login is temporarily unavailable.") from exc


def refresh_user(user_id: int) -> dict[str, Any] | None:
    try:
        initialize_database()
        with database_transaction() as connection:
            user = fetch_user_by_id(connection, int(user_id))
        if user is None or not user["is_active"]:
            return None
        return public_user(user)
    except sqlite3.Error:
        logger.exception("Could not refresh authenticated user user_id=%s", user_id)
        return None


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": int(user["user_id"]),
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
        "phone": user.get("phone"),
    }


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return (
        f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}$"
        f"{base64.b64encode(salt).decode('ascii')}$"
        f"{base64.b64encode(digest).decode('ascii')}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, iterations_text, salt_text, digest_text = password_hash.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        iterations = int(iterations_text)
        salt = base64.b64decode(salt_text.encode("ascii"), validate=True)
        expected_digest = base64.b64decode(digest_text.encode("ascii"), validate=True)
    except Exception:
        return False

    actual_digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual_digest, expected_digest)


def _validate_email(value: str) -> str:
    normalized = " ".join(str(value or "").split()).lower()
    if not normalized:
        raise ValidationError("Email address is required.")
    if "@" not in normalized or "." not in normalized.rsplit("@", 1)[-1]:
        raise ValidationError("Email address is invalid.")
    if len(normalized) > 120:
        raise ValidationError("Email address must be 120 characters or fewer.")
    return normalized
