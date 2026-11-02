import os
import logging
from langchain_core.tools import tool

from data.repositories import (
    get_user,
    get_user_id_by_identifier,
    get_reservations,
    update_subscription,
    VALID_ACTIONS,
)

logger = logging.getLogger(__name__)

_SOLUTION_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CULTPASS_DB = os.path.join(_SOLUTION_ROOT, "data", "external", "cultpass.db")


@tool
def get_user_account(user_identifier: str) -> str:
    """Retrieve CultPass account and subscription details for a user.

    Args:
        user_identifier: The user's CultPass user ID or registered email address.

    Returns:
        Account status, subscription tier, monthly quota, and member-since date.
    """
    try:
        account = get_user(_CULTPASS_DB, user_identifier)
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:
        logger.error("Account lookup error: %s", e)
        return f"Error looking up account: {e}"

    if account is None:
        return f"No user found with ID or email: {user_identifier}"

    status_label = "BLOCKED — cannot access services" if account.is_blocked else "Active"
    return (
        f"**User:** {account.full_name} (ID: {account.user_id})\n"
        f"**Email:** {account.email}\n"
        f"**Account status:** {status_label}\n"
        f"**Subscription:** {account.subscription_status or 'None'} ({account.tier or 'N/A'} tier)\n"
        f"**Monthly quota:** {account.monthly_quota or 'N/A'} experiences\n"
        f"**Member since:** {account.member_since or 'N/A'}"
    )


@tool
def get_user_reservations(user_identifier: str) -> str:
    """Retrieve a CultPass user's experience reservations.

    Args:
        user_identifier: The user's CultPass user ID or registered email address.

    Returns:
        List of upcoming and recent reservations with experience details.
    """
    try:
        user_id = get_user_id_by_identifier(_CULTPASS_DB, user_identifier)
        if user_id is None:
            return f"No user found with ID or email: {user_identifier}"
        reservations = get_reservations(_CULTPASS_DB, user_id)
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:
        logger.error("Reservation lookup error: %s", e)
        return f"Error looking up reservations: {e}"

    if not reservations:
        return f"No reservations found for user: {user_identifier}"

    lines = [f"**Reservations for {user_identifier}:**"]
    for r in reservations:
        lines.append(
            f"- [{r.status.upper()}] **{r.experience_title}** @ {r.location} on {r.when} (ID: {r.reservation_id})"
        )
    return "\n".join(lines)


@tool
def manage_subscription(user_identifier: str, action: str) -> str:
    """Cancel or pause a CultPass user's subscription.

    Args:
        user_identifier: The user's CultPass user ID or email.
        action: Either 'cancel' or 'pause'.

    Returns:
        Confirmation message or error.
    """
    action = action.lower().strip()
    if action not in VALID_ACTIONS:
        return f"Invalid action '{action}'. Accepted values are: {', '.join(VALID_ACTIONS)}."

    try:
        user_id = get_user_id_by_identifier(_CULTPASS_DB, user_identifier)
        if user_id is None:
            return f"No user found with ID or email: {user_identifier}"
        result = update_subscription(_CULTPASS_DB, user_id, action)
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:
        logger.error("Subscription management error: %s", e)
        return f"Error managing subscription: {e}"

    if not result.success:
        return result.message

    logger.info("Subscription %sed for user: %s", action, user_identifier)
    return (
        f"Subscription successfully **{action}d** for {user_identifier}. "
        f"Status is now '{result.new_status}'."
    )
