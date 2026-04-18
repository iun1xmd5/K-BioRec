"""
HKB-BV Backend Configuration
Production and development environment settings
"""

import os
from datetime import timedelta
from pathlib import Path

# ============================================================
# Base Directory
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent


def _ensure_dir(path: Path) -> Path:
    """Create directory and all parents if they do not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# Base Configuration
# ============================================================
class BaseConfig:
    """Shared configuration for all environments"""

    # ---- Application ----------------------------------------
    APP_NAME    = 'HKB-BV Biometric Verification API'
    APP_VERSION = '1.0.0'
    API_PREFIX  = '/api/v1'

    # ---- Security -------------------------------------------
    SECRET_KEY     = os.getenv('SECRET_KEY',     'change-me-in-production')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-change-me')
    JWT_ALGORITHM  = 'HS256'
    JWT_ACCESS_TOKEN_EXPIRES  = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7)

    # ---- Database -------------------------------------------
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO                = False
    DB_POOL_SIZE                   = 10
    DB_MAX_OVERFLOW                = 20
    DB_POOL_TIMEOUT                = 30
    DB_POOL_RECYCLE                = 3600

    # ---- MQTT / TLS -----------------------------------------
    MQTT_HOST        = os.getenv('MQTT_HOST', 'mqtt.hkb-bv.local')
    MQTT_PORT        = int(os.getenv('MQTT_PORT', 8883))
    MQTT_USER        = os.getenv('MQTT_USER', 'esp32_client')
    MQTT_PASS        = os.getenv('MQTT_PASS', '')
    MQTT_TOPIC_PROBE  = 'hkb_bv/probe'
    MQTT_TOPIC_RESULT = 'hkb_bv/result'
    MQTT_TOPIC_STATUS = 'hkb_bv/status'

    # ---- NIA Gateway ----------------------------------------
    NIDA_API_URL  = os.getenv('NIDA_API_URL',  'https://api.nida.go.tz/v1')
    NIIMS_API_URL = os.getenv('NIIMS_API_URL', 'https://api.niims.go.ke/v1')
    NIRA_API_URL  = os.getenv('NIRA_API_URL',  'https://api.nira.go.ug/v1')
    NIA_API_TIMEOUT = int(os.getenv('NIA_API_TIMEOUT', 10))
    NIA_API_RETRIES = int(os.getenv('NIA_API_RETRIES', 3))

    # ---- TLS Certificates -----------------------------------
    TLS_CA_CERT     = os.getenv('TLS_CA_CERT',     'certs/ca.crt')
    TLS_CLIENT_CERT = os.getenv('TLS_CLIENT_CERT', 'certs/client.crt')
    TLS_CLIENT_KEY  = os.getenv('TLS_CLIENT_KEY',  'certs/client.key')

    # ---- FAISS Indexing -------------------------------------
    FAISS_INDEX_PATH    = os.getenv(
        'FAISS_INDEX_PATH',
        str(BASE_DIR / 'data' / 'indices' / 'nia_gallery.faiss')
    )
    FAISS_EMBEDDING_DIM = 512
    FAISS_NPROBE        = 8
    FAISS_TOP_K         = 1

    # ---- HKB-BV Core Parameters -----------------------------
    DS_FUSION_THRESHOLD  = 0.65
    LIVENESS_THRESHOLD   = 0.65
    CONFLICT_THRESHOLD_K = 0.50
    ALPHA_SECURITY       = 0.70
    BETA_LATENCY         = 0.10
    LAMBDA_FUSION        = 0.30
    FAR_EPSILON          = 0.001
    LATENCY_MAX_MS       = 10

    # ---- SWRL Rules -----------------------------------------
    SWRL_RULES_FILE = os.getenv(
        'SWRL_RULES_FILE',
        str(BASE_DIR / 'configs' / 'swrl_rules.xml')
    )

    # ---- Logging --------------------------------------------
    LOG_LEVEL     = os.getenv('LOG_LEVEL',     'INFO')
    LOG_FILE      = os.getenv('LOG_FILE',      'logs/hkb_bv.log')
    AUDIT_LOG_FILE = os.getenv('AUDIT_LOG_FILE', 'logs/audit.log')

    # ---- Pagination -----------------------------------------
    DEFAULT_PAGE_SIZE = 50
    MAX_PAGE_SIZE     = 200

    # ---- PDPA Compliance ------------------------------------
    AUDIT_LOG_RETENTION_DAYS = 2555
    STORE_RAW_FINGERPRINT    = False
    HASH_ALGORITHM           = 'argon2'


# ============================================================
# Development Configuration
# ============================================================
class DevelopmentConfig(BaseConfig):
    """Development — SQLite in data/ directory"""

    DEBUG   = True
    TESTING = False

    # Ensure data directory exists before setting URI
    _data_dir = _ensure_dir(BASE_DIR / 'data')

    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DEV_DATABASE_URL',
        f"sqlite:///{_data_dir / 'hkb_bv_dev.db'}"
    )
    SQLALCHEMY_ECHO          = False
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    LOG_LEVEL                = 'DEBUG'
    NIA_API_TIMEOUT          = 30


# ============================================================
# Testing Configuration
# ============================================================
class TestingConfig(BaseConfig):
    """Testing — in-memory SQLite"""

    DEBUG   = False
    TESTING = True

    SQLALCHEMY_DATABASE_URI  = 'sqlite:///:memory:'
    SQLALCHEMY_ECHO          = False
    JWT_ACCESS_TOKEN_EXPIRES = False
    WTF_CSRF_ENABLED         = False
    NIA_API_TIMEOUT          = 5


# ============================================================
# Production Configuration
# ============================================================
class ProductionConfig(BaseConfig):
    """Production — PostgreSQL"""

    DEBUG   = False
    TESTING = False

    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'postgresql://hkb_bv_user:password@localhost:5432/hkb_bv_db'
    )
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    LOG_LEVEL                = 'WARNING'

    PREFERRED_URL_SCHEME    = 'https'
    SESSION_COOKIE_SECURE   = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'


# ============================================================
# Configuration Registry
# ============================================================
CONFIG_MAP = {
    'development': DevelopmentConfig,
    'testing':     TestingConfig,
    'production':  ProductionConfig,
    'default':     DevelopmentConfig,
}


def get_config(env: str = None) -> BaseConfig:
    """
    Return the appropriate configuration class.

    Args:
        env: 'development' | 'testing' | 'production'

    Returns:
        Configuration instance with all directories pre-created.
    """
    env          = env or os.getenv('FLASK_ENV', 'development')
    config_class = CONFIG_MAP.get(env, DevelopmentConfig)
    config       = config_class()

    # Ensure all required runtime directories exist
    _ensure_dir(BASE_DIR / 'data')
    _ensure_dir(BASE_DIR / 'data' / 'indices')
    _ensure_dir(BASE_DIR / 'logs')
    _ensure_dir(BASE_DIR / 'certs')
    _ensure_dir(BASE_DIR / 'checkpoints')

    return config
