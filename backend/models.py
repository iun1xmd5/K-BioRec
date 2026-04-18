"""
HKB-BV SQLAlchemy Database Models
PDPA-compliant schema: no raw biometric data stored
"""

import hashlib
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

db  = SQLAlchemy()
ph  = PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=2,
    hash_len=32,
    salt_len=16
)


# ============================================================
# User Model
# ============================================================
class User(db.Model):
    """
    System user (PSRS administrator, operator, auditor).
    Passwords are Argon2-hashed; plaintext is never stored.
    """
    __tablename__ = 'users'

    id         = db.Column(db.Integer,     primary_key=True)
    username   = db.Column(db.String(80),  unique=True,  nullable=False)
    email      = db.Column(db.String(120), unique=True,  nullable=False)
    role       = db.Column(
        db.String(20),
        nullable=False,
        default='operator'
    )  # 'admin', 'operator', 'auditor'
    password_hash = db.Column(db.String(256), nullable=False)
    is_active  = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # Relationship to audit logs
    audit_logs = db.relationship(
        'VerificationLog',
        backref='verified_by',
        lazy=True
    )

    def set_password(self, plaintext: str):
        """Hash and store password using Argon2"""
        self.password_hash = ph.hash(plaintext)

    def check_password(self, plaintext: str) -> bool:
        """Verify password against stored Argon2 hash"""
        try:
            return ph.verify(self.password_hash, plaintext)
        except VerifyMismatchError:
            return False
        except Exception:
            return False

    def to_dict(self) -> dict:
        return {
            'id':         self.id,
            'username':   self.username,
            'email':      self.email,
            'role':       self.role,
            'is_active':  self.is_active,
            'created_at': self.created_at.isoformat()
            if self.created_at else None
        }

    def __repr__(self):
        return f'<User {self.username} [{self.role}]>'


# ============================================================
# Biometric Template Model
# ============================================================
class BiometricTemplate(db.Model):
    """
    Hashed fingerprint embedding storage.
    Raw images are NEVER stored (PDPA compliance).
    Only SHA-256 hash of Argon2-hashed embeddings is persisted.
    """
    __tablename__ = 'biometric_templates'

    id                = db.Column(db.Integer,     primary_key=True)
    candidate_id      = db.Column(db.String(100), nullable=False, index=True)
    embedding_hash    = db.Column(db.String(256), nullable=False)  # SHA-256
    template_version  = db.Column(db.String(20),  default='1.0')
    sensor_model      = db.Column(db.String(50),  nullable=True)   # e.g., R307
    enrolment_centre  = db.Column(db.String(100), nullable=True)
    created_at        = db.Column(db.DateTime,    default=datetime.utcnow)
    is_active         = db.Column(db.Boolean,     default=True)

    def to_dict(self) -> dict:
        return {
            'id':               self.id,
            'candidate_id':     self.candidate_id,
            'template_version': self.template_version,
            'sensor_model':     self.sensor_model,
            'enrolment_centre': self.enrolment_centre,
            'created_at':       self.created_at.isoformat()
            if self.created_at else None
        }


# ============================================================
# Verification Log Model (Immutable Audit Trail)
# ============================================================
class VerificationLog(db.Model):
    """
    Immutable audit log for every verification event.
    Supports PDPA right-to-explanation and regulatory compliance.
    Records are append-only; no UPDATE or DELETE permitted.
    """
    __tablename__ = 'verification_logs'

    id                      = db.Column(db.Integer,    primary_key=True)
    candidate_id            = db.Column(db.String(100), nullable=False, index=True)
    session_id              = db.Column(db.String(64),  nullable=True,  index=True)

    # Biometric scores (no raw data)
    fingerprint_hash        = db.Column(db.String(256), nullable=True)
    similarity_score        = db.Column(db.Float,       nullable=True)
    liveness_score          = db.Column(db.Float,       nullable=True)
    final_belief            = db.Column(db.Float,       nullable=True)

    # Decision
    decision                = db.Column(db.Integer,     nullable=False)  # 0/1
    decision_threshold      = db.Column(db.Float,       default=0.65)
    conflict_k              = db.Column(db.Float,       nullable=True)

    # SWRL / KB reasoning trace
    fraud_flags             = db.Column(db.Text,        nullable=True)   # JSON string
    swrl_rules_fired        = db.Column(db.Text,        nullable=True)   # JSON string
    severity                = db.Column(db.String(20),  nullable=True)

    # Context
    recruitment_centre      = db.Column(db.String(100), nullable=True)
    device_id               = db.Column(db.String(100), nullable=True)
    firmware_version        = db.Column(db.String(20),  nullable=True)
    network_condition       = db.Column(db.String(50),  nullable=True)

    # Latency
    edge_latency_ms         = db.Column(db.Float,       nullable=True)
    backend_latency_ms      = db.Column(db.Float,       nullable=True)
    total_latency_ms        = db.Column(db.Float,       nullable=True)

    # Audit metadata
    timestamp               = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    user_id                 = db.Column(
        db.Integer,
        db.ForeignKey('users.id'),
        nullable=True
    )

    def to_dict(self) -> dict:
        import json
        return {
            'id':                   self.id,
            'candidate_id':         self.candidate_id,
            'session_id':           self.session_id,
            'similarity_score':     self.similarity_score,
            'liveness_score':       self.liveness_score,
            'final_belief':         self.final_belief,
            'decision':             self.decision,
            'decision_threshold':   self.decision_threshold,
            'conflict_k':           self.conflict_k,
            'fraud_flags':          json.loads(self.fraud_flags)
                                    if self.fraud_flags else [],
            'swrl_rules_fired':     json.loads(self.swrl_rules_fired)
                                    if self.swrl_rules_fired else [],
            'severity':             self.severity,
            'recruitment_centre':   self.recruitment_centre,
            'device_id':            self.device_id,
            'total_latency_ms':     self.total_latency_ms,
            'timestamp':            self.timestamp.isoformat()
                                    if self.timestamp else None
        }

    def __repr__(self):
        return (
            f'<VerificationLog candidate={self.candidate_id} '
            f'decision={self.decision} ts={self.timestamp}>'
        )
