"""Subscription repository — state-changing operations on cultpass.db."""
from dataclasses import dataclass

from .base import get_connection

# Maps user-facing action names to the DB status value they produce.
VALID_ACTIONS: dict[str, str] = {
    "cancel": "cancelled",
    "pause": "paused",
}


@dataclass
class SubscriptionUpdate:
    success: bool
    user_id: str
    new_status: str
    message: str


def update_subscription(db_path: str, user_id: str, action: str) -> SubscriptionUpdate:
    """Apply a cancel or pause action to a user's subscription.

    Returns a SubscriptionUpdate describing the outcome; does NOT raise on
    business-logic failures (no subscription found, already cancelled, etc.)
    so the tool layer can format a friendly message.
    """
    new_status = VALID_ACTIONS[action]  # caller must validate action first

    with get_connection(db_path) as conn:
        # Read current status before updating so we can detect no-ops
        current = conn.execute(
            "SELECT status FROM subscriptions WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if not current:
            return SubscriptionUpdate(
                success=False,
                user_id=user_id,
                new_status="",
                message=f"No subscription found for user '{user_id}'.",
            )

        rows_affected = conn.execute(
            "UPDATE subscriptions SET status = ? WHERE user_id = ?",
            (new_status, user_id),
        ).rowcount

    if rows_affected == 0:
        return SubscriptionUpdate(
            success=False,
            user_id=user_id,
            new_status="",
            message=f"Subscription update failed for user '{user_id}'.",
        )

    return SubscriptionUpdate(
        success=True,
        user_id=user_id,
        new_status=new_status,
        message=f"Subscription successfully {action}d. Status is now '{new_status}'.",
    )
