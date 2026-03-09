"""Hybrid Anthropic token resolution with fallback chain.

Fallback order:
1. MEM0_ANTHROPIC_TOKEN env var
2. ~/.claude/.credentials.json (claudeAiOauth.accessToken)
3. ANTHROPIC_API_KEY env var
4. None (disabled with warning)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import httpx

from mem0_mcp_selfhosted.env import opt_env

logger = logging.getLogger(__name__)

_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
_TOKEN_CACHE_DIR = Path.home() / ".cache" / "mem0-mcp"

# Anthropic OAuth endpoint and Claude Code's public client_id.
_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"


def is_oat_token(token: str) -> bool:
    """Detect whether the token is an OAT token (vs standard API key)."""
    return "sk-ant-oat" in token


def _read_credentials_file() -> str | None:
    """Read accessToken from ~/.claude/.credentials.json.

    Returns the token string or None. Handles:
    - Missing file (silent — file absence is expected)
    - Malformed JSON (warns)
    - Missing accessToken key (warns)
    """
    if not _CREDENTIALS_PATH.exists():
        return None

    try:
        data = json.loads(_CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse credentials file %s: %s", _CREDENTIALS_PATH, exc)
        return None

    try:
        token = data["claudeAiOauth"]["accessToken"]
    except (KeyError, TypeError):
        logger.warning(
            "Credentials file %s missing claudeAiOauth.accessToken", _CREDENTIALS_PATH
        )
        return None

    if not token or not isinstance(token, str):
        logger.warning("Credentials file accessToken is empty or invalid")
        return None

    return token


_DEFAULT_API_KEY_HELPER = Path.home() / ".wise-claude-code" / "token.sh"


def _jwt_exp(token: str) -> int | None:
    """Extract the ``exp`` claim (epoch seconds) from a JWT without verification.

    Returns None if the token is not a valid JWT or has no ``exp`` claim.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    # Base64url decode the payload (add padding as needed)
    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None
    exp = payload.get("exp")
    return int(exp) if isinstance(exp, (int, float)) else None


def _cache_path(script_path: str) -> Path:
    """Return the cache file path for a given helper script path."""
    key = hashlib.sha256(script_path.encode()).hexdigest()[:16]
    return _TOKEN_CACHE_DIR / f"token-{key}.json"


def _read_cached_token(script_path: str) -> str | None:
    """Return a cached token if it exists and is not expiring within 5 minutes."""
    try:
        data = json.loads(_cache_path(script_path).read_text(encoding="utf-8"))
        token = data.get("token", "")
        exp = data.get("exp")
        if not token or not isinstance(exp, (int, float)):
            return None
        # Reject if expiring within 300 seconds
        if int(exp) - int(time.time()) < 300:
            return None
        return token
    except Exception:
        return None


def _invalidate_cached_token(script_path: str) -> None:
    """Delete the cache file for a given helper script path."""
    try:
        _cache_path(script_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Failed to invalidate token cache: %s", exc)


def _write_cached_token(script_path: str, token: str) -> None:
    """Write a token + its JWT exp to the cache file."""
    exp = _jwt_exp(token)
    if exp is None:
        return  # Not a JWT — don't cache
    try:
        _TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(script_path).write_text(
            json.dumps({"token": token, "exp": exp}), encoding="utf-8"
        )
    except Exception as exc:
        logger.debug("Failed to write token cache: %s", exc)


def _run_api_key_helper(script_path: str) -> str | None:
    """Run an apiKeyHelper script and return its stdout as the token.

    Caches the JWT to disk and skips the script on subsequent calls while
    the token remains valid (with a 5-minute buffer before expiry).

    The script is executed via the shell. Its stdout is used as the token.
    Returns None on any error (missing file, non-zero exit, empty output).
    """
    path = Path(script_path).expanduser()
    if not path.exists():
        return None

    # Return cached token if still valid
    cached = _read_cached_token(script_path)
    if cached:
        logger.debug("Auth resolved from token cache (script: %s)", script_path)
        return cached

    try:
        result = subprocess.run(
            str(path),
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("apiKeyHelper script failed: %s", exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "apiKeyHelper script exited %d: %s", result.returncode, result.stderr[:200]
        )
        return None

    token = result.stdout.strip()
    if not token:
        logger.warning("apiKeyHelper script produced empty output")
        return None

    _write_cached_token(script_path, token)
    return token


def resolve_token(invalidate_cache: bool = False) -> str | None:
    """Resolve an Anthropic auth token using the prioritized fallback chain.

    Args:
        invalidate_cache: If True, delete any cached helper-script token before
            resolving, forcing the helper script to run and produce a fresh token.
            Use this when a 401 indicates the current token has expired.

    Returns the resolved token or None if no auth is available.
    """
    # Priority 1: Explicit env var
    token = opt_env("MEM0_ANTHROPIC_TOKEN")
    if token:
        token_type = "OAT" if is_oat_token(token) else "API key"
        logger.debug("Auth resolved from MEM0_ANTHROPIC_TOKEN (type: %s)", token_type)
        return token

    # Priority 2: Claude Code credentials file
    token = _read_credentials_file()
    if token:
        token_type = "OAT" if is_oat_token(token) else "API key"
        logger.debug(
            "Auth resolved from %s (type: %s)", _CREDENTIALS_PATH, token_type
        )
        return token

    # Priority 2.5: apiKeyHelper script (e.g. Wise gateway JWT)
    helper_path = opt_env("MEM0_API_KEY_HELPER")
    if not helper_path and _DEFAULT_API_KEY_HELPER.exists():
        helper_path = str(_DEFAULT_API_KEY_HELPER)
    if helper_path:
        if invalidate_cache:
            _invalidate_cached_token(helper_path)
        token = _run_api_key_helper(helper_path)
        if token:
            logger.debug("Auth resolved from apiKeyHelper script: %s", helper_path)
            return token

    # Priority 3: Standard API key
    token = opt_env("ANTHROPIC_API_KEY")
    if token:
        token_type = "OAT" if is_oat_token(token) else "API key"
        logger.debug("Auth resolved from ANTHROPIC_API_KEY (type: %s)", token_type)
        return token

    # No auth available
    logger.warning(
        "No Anthropic token found. Checked: MEM0_ANTHROPIC_TOKEN env var, "
        "%s, apiKeyHelper script, ANTHROPIC_API_KEY env var. "
        "Anthropic LLM features will be disabled.",
        _CREDENTIALS_PATH,
    )
    return None


def read_credentials_full() -> dict | None:
    """Read the full claudeAiOauth object from credentials file.

    Returns dict with access_token, refresh_token, expires_at, scopes
    or None if the file is missing, malformed, or lacks required fields.
    """
    if not _CREDENTIALS_PATH.exists():
        return None

    try:
        data = json.loads(_CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse credentials file %s: %s", _CREDENTIALS_PATH, exc)
        return None

    oauth = data.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None

    access_token = oauth.get("accessToken")
    refresh_token = oauth.get("refreshToken")
    expires_at = oauth.get("expiresAt")

    if not access_token or not isinstance(access_token, str):
        return None
    if not refresh_token or not isinstance(refresh_token, str):
        return None
    if expires_at is not None and not isinstance(expires_at, (int, float)):
        logger.warning("Credentials file expiresAt is not a number, ignoring")
        expires_at = None

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "scopes": oauth.get("scopes"),
    }


def refresh_oat_token(refresh_token: str) -> dict | None:
    """Exchange a refresh token for a new access token via Anthropic OAuth.

    Returns dict with access_token, refresh_token, expires_in on success.
    Returns None on any failure (invalid token, network error, non-200).
    Never writes to disk — caller stores tokens in memory.
    """
    try:
        response = httpx.post(
            _OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _OAUTH_CLIENT_ID,
            },
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        logger.warning("[mem0] OAuth refresh network error: %s", exc)
        return None

    if response.status_code == 200:
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("[mem0] OAuth refresh returned invalid JSON: %s", exc)
            return None

        access_token = data.get("access_token")
        new_refresh_token = data.get("refresh_token")
        if not access_token or not new_refresh_token:
            logger.warning("[mem0] OAuth refresh response missing required fields")
            return None

        logger.info("[mem0] OAT token refreshed via OAuth endpoint")
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "expires_in": data.get("expires_in"),
        }

    if response.status_code in (400, 401):
        logger.warning(
            "[mem0] OAuth refresh failed (HTTP %d): refresh token invalid or consumed",
            response.status_code,
        )
        return None

    logger.warning("[mem0] OAuth refresh unexpected status: HTTP %d", response.status_code)
    return None


def is_token_expiring_soon(
    expires_at_ms: int | None, threshold_seconds: int = 1800
) -> bool:
    """Check whether an OAT token will expire within the given threshold.

    Args:
        expires_at_ms: Token expiry as epoch milliseconds, or None.
        threshold_seconds: How many seconds before expiry to consider "soon".

    Returns True if the token is expired or will expire within threshold.
    Returns False if expires_at_ms is None (cannot determine, assume valid).
    """
    if expires_at_ms is None:
        return False

    now_ms = int(time.time() * 1000)
    threshold_ms = threshold_seconds * 1000
    return (expires_at_ms - now_ms) < threshold_ms
