"""Reservation repository — booking lookups from cultpass.db."""
from dataclasses import dataclass
from typing import Optional

from .base import get_connection


@dataclass
class Reservation:
    reservation_id: str
    status: str
    experience_title: str
    location: str
    when: str


def get_user_id_by_identifier(db_path: str, identifier: str) -> Optional[str]:
    """Resolve a user_id or email to a canonical user_id. Returns None if not found."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ? OR email = ?",
            (identifier, identifier),
        ).fetchone()
    return row["user_id"] if row else None


def get_reservations(db_path: str, user_id: str) -> list[Reservation]:
    """Return all reservations for a user, ordered by experience date."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.reservation_id, r.status, e.title, e.location, e."when"
            FROM reservations r
            JOIN experiences e ON r.experience_id = e.experience_id
            WHERE r.user_id = ?
            ORDER BY e."when" ASC
            """,
            (user_id,),
        ).fetchall()

    return [
        Reservation(
            reservation_id=row["reservation_id"],
            status=row["status"],
            experience_title=row["title"],
            location=row["location"],
            when=str(row["when"]),
        )
        for row in rows
    ]
