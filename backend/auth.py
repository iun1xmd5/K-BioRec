"""
HKB-BV Authentication Module
OAuth 2.0 / JWT token management and role-based access control
"""

import os
import logging
from functools import wraps
from datetime import datetime

from flask import jsonify, request
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    verify_jwt_in_request
)

logger = logging.getLogger(__name__)

# Initialise JWT manager (bound to app in app.py)
jwt = JWTManager()

# ============================================================
# Role Definitions
# ============================================================
ROLES = {
    'admin':    ['read', 'write', 'delete', 'audit', 'admin'],
    'auditor':  ['read', 'audit'],
    'operator': ['read', 'write'],
}


# ============================================================
# Role-Based Access Control Decorator
# ============================================================
def require_role(*allowed_roles):
    """
    Decorator: restrict endpoint to users with specified roles.

    Usage:
        @require_role('admin', 'auditor')
        def my_endpoint(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                verify_jwt_in_request()
                claims = get_jwt()
                user_role = claims.get('role', '')

                if user_role not in allowed_roles:
                    logger.warning(
                        f"Access denied: role='{user_role}' "
                        f"required={allowed_roles}"
                    )
                    return jsonify({
                        'error': 'Access denied. Insufficient privileges.'
                    }), 403

                return fn(*args, **kwargs)

            except Exception as e:
                logger.error(f"Auth error: {str(e)}")
                return jsonify({'error': 'Authentication failed'}), 401

        return wrapper
    return decorator


# ============================================================
# Token Generation Helpers
# ============================================================
def generate_tokens(user_id: int, role: str) -> dict:
    """
    Generate access and refresh JWT tokens for a user.

    Args:
        user_id: Database user ID
        role: User role string

    Returns:
        {'access_token': str, 'refresh_token': str}
    """
    additional_claims = {'role': role}

    access_token = create_access_token(
        identity=user_id,
        additional_claims=additional_claims
    )
    refresh_token = create_refresh_token(
        identity=user_id,
        additional_claims=additional_claims
    )

    logger.info(f"Tokens generated for user_id={user_id}, role={role}")

    return {
        'access_token':  access_token,
        'refresh_token': refresh_token,
        'token_type':    'Bearer'
    }


# ============================================================
# JWT Error Handlers (registered in app.py)
# ============================================================
def register_jwt_handlers(jwt_manager: JWTManager):
    """Register all JWT error callback handlers."""

    @jwt_manager.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'error': 'Token has expired. Please log in again.'
        }), 401

    @jwt_manager.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({
            'error': f'Invalid token: {error}'
        }), 401

    @jwt_manager.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({
            'error': 'Authentication token required.'
        }), 401

    @jwt_manager.revoked_token_loader
    def revoked_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'error': 'Token has been revoked.'
        }), 401
