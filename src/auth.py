"""Password hashing and opaque, server-side login sessions."""
import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from src import database

SESSION_COOKIE = "beanguard_session"
SESSION_DAYS = 7
PASSWORD_ITERATIONS = 310_000


def _secret_key():
    secret = os.getenv("APP_SECRET_KEY", "")
    if len(secret.encode("utf-8")) < 32:
        raise RuntimeError("APP_SECRET_KEY must contain at least 32 characters.")
    return secret.encode("utf-8")


def validate_configuration():
    _secret_key()


def _token_digest(token):
    return hmac.new(_secret_key(), token.encode("utf-8"), hashlib.sha256).hexdigest()


def hash_password(password):
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${derived.hex()}"


def verify_password(password, encoded):
    try:
        algorithm, iterations, salt_hex, expected_hex = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual.hex(), expected_hex)


def create_account(email, display_name, password):
    email = email.strip().lower()
    display_name = display_name.strip()
    if len(email) > 254 or email.count("@") != 1 or "." not in email.rsplit("@", 1)[1]:
        raise ValueError("Enter a valid email address.")
    if not display_name or len(display_name) > 80:
        raise ValueError("Name must be between 1 and 80 characters.")
    if len(password) < 10 or len(password) > 128:
        raise ValueError("Password must be between 10 and 128 characters.")
    if database.user_by_email(email):
        raise ValueError("An account with that email already exists.")
    try:
        user_id = database.create_user(email, display_name, hash_password(password))
    except sqlite3.IntegrityError as error:
        raise ValueError("An account with that email already exists.") from error
    return database.user_by_id(user_id)


def authenticate(email, password):
    if not password or len(password) > 128:
        return None
    user = database.user_by_email(email.strip().lower())
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return {key: user[key] for key in ("id", "email", "display_name")}


def create_session(user_id):
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)).isoformat()
    database.add_session(_token_digest(token), user_id, expires_at)
    return token


def get_session_user(token):
    if not token:
        return None
    current_time = datetime.now(timezone.utc).isoformat()
    return database.session_user(_token_digest(token), current_time)


def delete_session(token):
    if token:
        database.remove_session(_token_digest(token))
