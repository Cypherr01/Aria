"""
api.middleware.auth
====================
Simple Bearer-token API key authentication for ARIA endpoints.

Development mode: if ARIA_API_KEY is not set in the environment,
all requests are allowed through (returns "dev_mode").
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    """
    Validate the Bearer token in the Authorization header.

    Returns:
        The raw token string on success, or ``"dev_mode"`` when no key is
        configured.

    Raises:
        HTTPException(401): When a key is configured but the provided token
                            does not match.
    """
    expected = os.getenv("ARIA_API_KEY", "")
    if not expected:
        return "dev_mode"
    if not credentials or credentials.credentials != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. "
                   "Provide a valid Bearer token in the Authorization header.",
        )
    return credentials.credentials
