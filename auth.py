"""
ShiftCover — Authentication module (Azure AD + local users).

Backends:
  1. Azure AD / Microsoft Entra ID (OIDC + PKCE)
  2. Local users file (YAML with bcrypt hashes) — always enabled as fallback

Configuration is read from auth_config.yaml and environment variables.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
import urllib.parse
from pathlib import Path
from typing import Any

import yaml
import streamlit as st

_AUTH_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _AUTH_DIR / "auth_config.yaml"

# ---------------------------------------------------------------------------
# Default session lifetime (8 hours)
# ---------------------------------------------------------------------------
_SESSION_SECONDS = 8 * 3600


# ===========================================================================
# Config loading
# ===========================================================================
def _load_config() -> dict[str, Any] | None:
    if not _CONFIG_PATH.exists():
        return None
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _env(key: str, default: str = "") -> str:
    """Read from env, then Streamlit secrets, then default."""
    val = os.getenv(key)
    if val:
        return val
    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass
    return default


# ===========================================================================
# Hashing helpers (SHA-256, no external deps needed)
# ===========================================================================
def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


# ===========================================================================
# Token helpers
# ===========================================================================
def _generate_token(nbytes: int = 32) -> str:
    return secrets.token_hex(nbytes)


def _store_token(name: str, value: str):
    st.session_state[f"_auth_{name}"] = value


def _pop_token(name: str) -> str | None:
    key = f"_auth_{name}"
    val = st.session_state.get(key)
    if key in st.session_state:
        del st.session_state[key]
    return val


# ===========================================================================
# Azure AD backend
# ===========================================================================
def _azure_discovery(tenant: str) -> dict | None:
    """Fetch OpenID Connect discovery document."""
    import json
    from urllib.request import urlopen

    url = f"https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration"
    try:
        with urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _azure_start_login(config: dict):
    """Redirect user to Microsoft login."""
    import base64
    import hashlib

    tenant = _env("AZURE_TENANT_ID", config.get("azure", {}).get("tenant_id", ""))
    client_id = _env("AZURE_CLIENT_ID", config.get("azure", {}).get("client_id", ""))
    redirect_uri = config.get("azure", {}).get("redirect_uri", "")
    if not redirect_uri:
        # Auto-detect: assume the Streamlit app URL + ?_auth_callback=azure
        redirect_uri = "http://localhost:8501/"

    discovery = _azure_discovery(tenant)
    if not discovery:
        raise RuntimeError("Cannot reach Azure AD discovery endpoint.")

    # PKCE
    code_verifier = _generate_token(43)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    state = _generate_token(16)
    nonce = _generate_token(16)

    _store_token("azure_code_verifier", code_verifier)
    _store_token("azure_state", state)
    _store_token("azure_nonce", nonce)
    _store_token("azure_redirect_uri", redirect_uri)

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": "openid profile email",
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = discovery["authorization_endpoint"] + "?" + urllib.parse.urlencode(params)

    # Streamlit doesn't support redirect in the usual sense; we use a meta-refresh
    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={auth_url}">',
        unsafe_allow_html=True,
    )
    st.markdown(f"[Click here if not redirected]({auth_url})")
    st.stop()


def _azure_handle_callback(config: dict):
    """Called after Azure redirects back with ?code=..."""
    import json
    from urllib.request import Request, urlopen

    tenant = _env("AZURE_TENANT_ID", config.get("azure", {}).get("tenant_id", ""))
    client_id = _env("AZURE_CLIENT_ID", config.get("azure", {}).get("client_id", ""))
    redirect_uri = _pop_token("azure_redirect_uri") or config.get("azure", {}).get("redirect_uri", "http://localhost:8501/")
    code_verifier = _pop_token("azure_code_verifier")
    saved_state = _pop_token("azure_state")
    saved_nonce = _pop_token("azure_nonce")

    qp = st.query_params
    code = qp.get("code")
    state = qp.get("state")
    # Clear query params so the URL is clean afterwards
    st.query_params.clear()

    if not code or not state:
        st.error("Authentication callback missing code or state.")
        st.stop()

    # Validate state (prevent CSRF)
    if saved_state is None or not _constant_time_compare(state, saved_state):
        st.error("Authentication state mismatch. Please try again.")
        st.stop()

    discovery = _azure_discovery(tenant)
    if not discovery:
        st.error("Cannot reach Azure AD token endpoint.")
        st.stop()

    token_url = discovery["token_endpoint"]
    body = urllib.parse.urlencode({
        "client_id": client_id,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier or "",
    }).encode()

    try:
        req = Request(token_url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urlopen(req, timeout=10) as resp:
            token_data = json.loads(resp.read())
    except Exception as exc:
        st.error(f"Token exchange failed: {exc}")
        st.stop()

    id_token = token_data.get("id_token", "")
    if not id_token:
        st.error("No id_token received from Azure AD.")
        st.stop()

    # Decode the ID token (without verifying signature for now – we trust the HTTPS token endpoint)
    try:
        import base64
        parts = id_token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT structure")
        # Add padding
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        st.error("Failed to parse ID token.")
        st.stop()

    # Validate nonce (prevent replay)
    if saved_nonce and payload.get("nonce") != saved_nonce:
        st.error("ID token nonce mismatch.")
        st.stop()

    email = payload.get("email") or payload.get("preferred_username") or payload.get("upn", "")
    name = payload.get("name") or payload.get("given_name", email)
    oid = payload.get("oid", "")

    # Map Azure groups to local roles (if group claims are present)
    role = _map_azure_role(payload, config)

    _set_session(name or email, email, role or "user", "azure", oid)


def _map_azure_role(payload: dict, config: dict) -> str:
    """Map Azure AD group/role claims to local role names."""
    role_mapping = config.get("azure", {}).get("role_mapping", {})
    groups = payload.get("groups", []) or []

    if isinstance(groups, list):
        for g in groups:
            mapped = role_mapping.get(g)
            if mapped:
                return mapped

    # Check 'roles' claim (app roles)
    roles_claim = payload.get("roles", []) or []
    if isinstance(roles_claim, list):
        for r in roles_claim:
            mapped = role_mapping.get(r)
            if mapped:
                return mapped

    # Default role for any authenticated user
    return config.get("azure", {}).get("default_role", "user")


# ===========================================================================
# Local users backend
# ===========================================================================
def _local_login(config: dict, username: str, password: str) -> bool:
    """Authenticate against local users in auth_config.yaml."""
    users = config.get("users", {})
    user_entry = users.get(username)
    if not user_entry:
        return False
    stored_hash = user_entry.get("password_hash", "")
    # Simple bcrypt-ish: we store SHA-256 hashes for simplicity
    # In production, use bcrypt via passlib; we support both formats
    if stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$"):
        # bcrypt
        try:
            import bcrypt
            if bcrypt.checkpw(password.encode(), stored_hash.encode()):
                role = user_entry.get("role", "user")
                display = user_entry.get("name", username)
                _set_session(display, username, role, "local")
                return True
        except ImportError:
            st.error("bcrypt is required for password verification. Install: pip install bcrypt")
            return False
    else:
        # SHA-256 fallback
        if _constant_time_compare(_sha256(password), stored_hash):
            role = user_entry.get("role", "user")
            display = user_entry.get("name", username)
            _set_session(display, username, role, "local")
            return True
    return False


# ===========================================================================
# Session management
# ===========================================================================
def _set_session(display_name: str, username: str, role: str, backend: str,
                 external_id: str = ""):
    st.session_state["_auth_authenticated"] = True
    st.session_state["_auth_user"] = display_name
    st.session_state["_auth_username"] = username
    st.session_state["_auth_role"] = role
    st.session_state["_auth_backend"] = backend
    st.session_state["_auth_external_id"] = external_id
    st.session_state["_auth_login_time"] = time.time()


def logout():
    """Clear authentication state."""
    keys = [k for k in st.session_state if k.startswith("_auth_")]
    for k in keys:
        del st.session_state[k]
    st.rerun()


def get_user() -> dict[str, Any] | None:
    """Return current authenticated user info, or None."""
    if not st.session_state.get("_auth_authenticated"):
        return None
    # Check session expiry
    login_time = st.session_state.get("_auth_login_time", 0)
    if time.time() - login_time > _SESSION_SECONDS:
        for k in list(st.session_state.keys()):
            if k.startswith("_auth_"):
                del st.session_state[k]
        return None
    return {
        "name": st.session_state.get("_auth_user", ""),
        "username": st.session_state.get("_auth_username", ""),
        "role": st.session_state.get("_auth_role", "user"),
        "backend": st.session_state.get("_auth_backend", ""),
        "external_id": st.session_state.get("_auth_external_id", ""),
    }


def require_role(*roles: str) -> bool:
    """Check if the current user has one of the required roles. Returns True if allowed."""
    user = get_user()
    if not user:
        return False
    if not roles:
        return True
    return user["role"] in roles


def is_admin() -> bool:
    return require_role("admin")


# ===========================================================================
# Login UI entry point
# ===========================================================================
def require_login():
    """Call at the top of your Streamlit app. Shows login page if not authenticated."""
    config = _load_config() or {}

    # Check for Azure AD callback
    if st.query_params.get("code"):
        _azure_handle_callback(config)
        st.rerun()

    user = get_user()
    if user:
        return  # already authenticated

    # ── Show login page ───────────────────────────────────────────────────
    st.set_page_config(page_title="Simplex – Login", layout="centered")
    st.title("🕐 Simplex")
    st.subheader("Sign in")

    azure_config = config.get("azure", {})
    azure_enabled = bool(
        azure_config.get("tenant_id") or _env("AZURE_TENANT_ID")
    ) and bool(
        azure_config.get("client_id") or _env("AZURE_CLIENT_ID")
    )

    local_enabled = bool(config.get("users"))

    if azure_enabled:
        st.markdown("---")
        if st.button("🔐 Sign in with Microsoft (Azure AD)", type="primary",
                     use_container_width=True):
            _azure_start_login(config)

    if local_enabled and (
        not azure_enabled
        or st.checkbox("Sign in with local account instead", key="_show_local")
    ):
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary",
                                              use_container_width=True)
            if submitted:
                if _local_login(config, username, password):
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    if not azure_enabled and not local_enabled:
        st.info(
            "No authentication configured. To enable authentication:\n\n"
            "1. Copy `auth_config.example.yaml` to `auth_config.yaml`\n"
            "2. Edit the file with your users or Azure AD settings\n"
            "3. Set `AZURE_TENANT_ID` and `AZURE_CLIENT_ID` environment variables "
            "for Azure AD"
        )
        # Auto-login as guest if no auth configured
        _set_session("Guest", "guest", "admin", "none")
        st.rerun()

    st.stop()


# ===========================================================================
# Sidebar widget
# ===========================================================================
def render_sidebar_user():
    """Render a user-info / logout section in the sidebar."""
    user = get_user()
    if not user:
        return
    with st.sidebar:
        st.divider()
        role_badge = {
            "admin": "🔴 Admin",
            "user": "🟢 User",
            "viewer": "🔵 Viewer",
        }.get(user["role"], f"⚪ {user['role']}")
        st.caption(f"**{user['name']}**  ·  {role_badge}")
        if user["backend"] == "azure":
            st.caption(f"via Microsoft Azure AD")
        elif user["backend"] == "local":
            st.caption(f"via local account")
        if st.button("🚪 Sign out", use_container_width=True):
            logout()