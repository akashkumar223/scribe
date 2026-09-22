"""
Lightweight authentication — file-based user storage (survives restarts,
no database setup needed) and password hashing using Python's built-in
hashlib (no bcrypt/passlib dependency, avoids Windows build-tool issues
you've hit before with other packages).
"""
import os
import json
import hashlib
import secrets

USERS_FILE = os.getenv("USERS_FILE", "./users.json")

# Change this in your .env for a real deployment — anyone with this code
# can register as an admin. Treat it like a password.
ADMIN_INVITE_CODE = os.getenv("ADMIN_INVITE_CODE", "SCRIBE-ADMIN-2026")


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str = None) -> str:
    """Returns 'salt$hash' — salted PBKDF2 hash, no external crypto library needed."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    salt, _ = stored.split("$")
    return _hash_password(password, salt) == stored


def register_user(email: str, password: str, role: str, admin_code: str | None) -> dict:
    email = email.strip().lower()
    users = _load_users()

    if email in users:
        raise ValueError("An account with this email already exists.")

    if role == "admin":
        if admin_code != ADMIN_INVITE_CODE:
            raise ValueError("Invalid admin invite code.")

    users[email] = {
        "email": email,
        "password_hash": _hash_password(password),
        "role": role,
    }
    _save_users(users)
    return {"email": email, "role": role}


def login_user(email: str, password: str) -> dict:
    email = email.strip().lower()
    users = _load_users()

    user = users.get(email)
    if not user or not _verify_password(password, user["password_hash"]):
        raise ValueError("Invalid email or password.")

    return {"email": user["email"], "role": user["role"]}