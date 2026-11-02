"""Account repository — user and subscription lookups from cultpass.db."""
from dataclasses import dataclass
from typing import Optional

from .base import get_connection


@dataclass
class UserAccount:
    user_id: str
    full_name: str
    email: str
    is_blocked: bool
    subscription_status: Optional[str]
    tier: Optional[str]
    monthly_quota: Optional[int]
    member_since: Optional[str]


def get_user(db_path: str, identifier: str) -> Optional[UserAccount]:
    """Look up a user by user_id or email. Returns None if not found."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT u.user_id, u.full_name, u.email, u.is_blocked,
                   s.status, s.tier, s.monthly_quota, s.started_at
            FROM users u
            LEFT JOIN subscriptions s ON u.user_id = s.user_id
            WHERE u.user_id = ? OR u.email = ?
            """,
            (identifier, identifier),
        ).fetchone()

    if not row:
        return None

    return UserAccount(
        user_id=row["user_id"],
        full_name=row["full_name"],
        email=row["email"],
        is_blocked=bool(row["is_blocked"]),
        subscription_status=row["status"],
        tier=row["tier"],
        monthly_quota=row["monthly_quota"],
        member_since=row["started_at"],
    )
