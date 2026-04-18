"""
HKB-BV Flask REST API — Complete Application
Biometric verification pipeline with DS fusion, SWRL ontology,
NIA gateway integration, and PDPA-compliant audit logging.

Endpoints:
    GET  /health
    POST /auth/register
    POST /auth/login
    POST /auth/refresh
    POST /api/v1/verify
    POST /api/v1/verify/batch
    POST /api/v1/enrol
    GET  /api/v1/audit-logs
    GET  /api/v1/dashboard/metrics
    GET  /api/v1/candidate/<id>/history

Run:
    python -m backend.app                  (development)
    gunicorn "backend.app:create_app()"    (production)
"""

import os
import json
import logging
import hashlib
import numpy as np
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_jwt_extended import (
    jwt_required,
    get_jwt_identity,
    get_jwt,
    create_access_token,
)
from dotenv import load_dotenv

# ============================================================
# Local imports
# ============================================================
from backend.config           import get_config
from backend.models           import db, User, VerificationLog, BiometricTemplate
from backend.auth             import (
    jwt,
    generate_tokens,
    require_role,
    register_jwt_handlers
)
from backend.dempster_shafer  import DempsterShaferFusion
from backend.swrl_ontology    import SWRLEngine
from backend.nia_gateway      import NIAGateway

# ============================================================
# Environment & Logging
# ============================================================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Application Factory
# ============================================================

# In backend/app.py � update create_app() to ensure dirs exist

def create_app(env: str = None) -> Flask:
    """Create and configure the Flask application."""
    
    app = Flask(__name__)

    # ---- Configuration ------------------------------------
    config = get_config(env or os.getenv('FLASK_ENV', 'development'))
    app.config.from_object(config)

    # ---- Ensure required directories exist ----------------
    _ensure_runtime_dirs()

    # ---- Extensions ---------------------------------------
    db.init_app(app)
    jwt.init_app(app)
    register_jwt_handlers(jwt)

    # ---- Service singletons -------------------------------
    with app.app_context():
        app.ds_fusion   = DempsterShaferFusion(
            lambda_param       = config.LAMBDA_FUSION,
            conflict_threshold = config.CONFLICT_THRESHOLD_K
        )
        app.swrl_engine = SWRLEngine(
            rules_file = config.SWRL_RULES_FILE
        )
        app.nia_gateway = NIAGateway(
            nida_api_url  = config.NIDA_API_URL,
            niims_api_url = config.NIIMS_API_URL,
            nira_api_url  = config.NIRA_API_URL,
            timeout       = config.NIA_API_TIMEOUT
        )
        app.hkb_config = config
        db.create_all()   # Creates tables if they do not exist

    _register_routes(app)

    logger.info(
        "HKB-BV API ready [env=%s | db=%s]",
        os.getenv('FLASK_ENV', 'development'),
        app.config.get('SQLALCHEMY_DATABASE_URI', '')[:50]
    )
    return app


def _ensure_runtime_dirs():
    """Create all required runtime directories."""
    from pathlib import Path
    
    base = Path(__file__).resolve().parent.parent
    
    required = [
        base / 'data',
        base / 'data' / 'indices',
        base / 'logs',
        base / 'certs',
        base / 'checkpoints',
        base / 'results',
    ]
    
    for directory in required:
        directory.mkdir(parents=True, exist_ok=True)

def _register_routes(app: Flask):  # noqa: C901
    """Attach all endpoint handlers to the application."""

    cfg = app.hkb_config   # shorthand used in closures below

    # ----------------------------------------------------------
    # HEALTH
    # ----------------------------------------------------------

    @app.route('/health', methods=['GET'])
    def health():
        """Public health-check endpoint."""
        return jsonify({
            'status':    'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version':   cfg.APP_VERSION,
            'services':  {
                'database':    _db_ping(app),
                'swrl_engine': 'operational',
                'nia_gateway': (
                    'reachable'
                    if app.nia_gateway.is_reachable()
                    else 'unreachable'
                )
            }
        }), 200

    # ----------------------------------------------------------
    # AUTH — REGISTER
    # ----------------------------------------------------------

    @app.route('/auth/register', methods=['POST'])
    def register():
        """Register a new system user (administrator / operator)."""
        data     = request.get_json() or {}
        username = data.get('username', '').strip()
        email    = data.get('email',    '').strip()
        password = data.get('password', '')
        role     = data.get('role', 'operator').strip()

        # Input validation
        if not username:
            return jsonify({'error': 'username is required'}), 400
        if not email:
            return jsonify({'error': 'email is required'}), 400
        if not password or len(password) < 8:
            return jsonify(
                {'error': 'password must be at least 8 characters'}
            ), 400
        if role not in ('admin', 'operator', 'auditor'):
            return jsonify(
                {'error': "role must be 'admin', 'operator', or 'auditor'"}
            ), 400

        # Duplicate check
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 409
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 409

        user = User(username=username, email=email, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        logger.info("User registered: %s [%s]", username, role)
        return jsonify({'message': 'User registered successfully'}), 201

    # ----------------------------------------------------------
    # AUTH — LOGIN
    # ----------------------------------------------------------

    @app.route('/auth/login', methods=['POST'])
    def login():
        """Authenticate user and return JWT access + refresh tokens."""
        data     = request.get_json() or {}
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'error': 'username and password required'}), 400

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401
        if not user.is_active:
            return jsonify({'error': 'Account is disabled'}), 401

        user.last_login = datetime.utcnow()
        db.session.commit()

        tokens = generate_tokens(user.id, user.role)
        logger.info("Login: %s", username)
        return jsonify(tokens), 200

    # ----------------------------------------------------------
    # AUTH — REFRESH
    # ----------------------------------------------------------

    @app.route('/auth/refresh', methods=['POST'])
    @jwt_required(refresh=True)
    def refresh():
        """Issue a new access token using the refresh token."""
        user_id = get_jwt_identity()
        claims  = get_jwt()
        role    = claims.get('role', 'operator')

        new_token = create_access_token(
            identity=user_id,
            additional_claims={'role': role}
        )
        return jsonify({'access_token': new_token}), 200

    # ----------------------------------------------------------
    # VERIFY — Single candidate
    # ----------------------------------------------------------

    @app.route('/api/v1/verify', methods=['POST'])
    @jwt_required()
    def verify():
        """
        Full HKB-BV verification pipeline.

        Pipeline:
            1. Liveness gate          (edge-computed b_l)
            2. 1:N NIA gallery match  (FAISS / NIA API)
            3. SWRL fraud inference   (contextual KB rules)
            4. DS evidence fusion     (b = DS(m_DL, m_KB) · b_l)
            5. Threshold decision     (ŷ = I{b > τ*})
            6. PDPA-compliant logging (immutable audit trail)

        Request JSON:
            fingerprint_embedding : list[float]  512-D vector
            liveness_score        : float         [0, 1]
            candidate_id          : str
            recruitment_centre    : str           (optional)

        Response JSON:
            decision, belief, confidence, similarity_score,
            liveness_score, fraud_flags, swrl_rules_fired,
            severity, timestamp, audit_log_id
        """
        start_ms = _now_ms()
        data     = request.get_json() or {}

        # ---- Parse request ----------------------------------
        embedding = data.get('fingerprint_embedding')
        liveness  = _safe_float(data.get('liveness_score', 0.0))
        cand_id   = data.get('candidate_id', 'UNKNOWN')
        centre    = data.get('recruitment_centre', 'Unknown')
        device_id = data.get('device_id', 'Unknown')

        if embedding is None:
            return jsonify({'error': 'fingerprint_embedding is required'}), 400

        if len(embedding) != cfg.FAISS_EMBEDDING_DIM:
            return jsonify({
                'error': (
                    f'fingerprint_embedding must have '
                    f'{cfg.FAISS_EMBEDDING_DIM} dimensions, '
                    f'got {len(embedding)}'
                )
            }), 400

        # ---- Step 1: Liveness gate --------------------------
        if liveness < cfg.LIVENESS_THRESHOLD:
            log = _save_log(
                app,
                candidate_id     = cand_id,
                similarity_score = 0.0,
                liveness_score   = liveness,
                final_belief     = 0.0,
                decision         = 0,
                conflict_k       = None,
                fraud_flags      = ['liveness_failed'],
                swrl_rules_fired = [],
                severity         = 'high',
                centre           = centre,
                device_id        = device_id,
                latency_ms       = _now_ms() - start_ms,
                user_id          = get_jwt_identity()
            )

            logger.warning(
                "Spoof rejected: cand=%s, liveness=%.3f",
                cand_id, liveness
            )

            return jsonify({
                'decision':         0,
                'belief':           0.0,
                'confidence':       0.0,
                'similarity_score': 0.0,
                'liveness_score':   round(liveness, 4),
                'fraud_flags':      ['liveness_failed'],
                'swrl_rules_fired': [],
                'severity':         'high',
                'reason':           'Liveness check failed — spoof detected',
                'timestamp':        datetime.utcnow().isoformat(),
                'audit_log_id':     log.id
            }), 200

        try:
            # ---- Step 2: 1:N NIA gallery match --------------
            sim_score, gallery_id = _query_nia_gallery(
                app, embedding, cand_id
            )

            # ---- Step 3: SWRL fraud inference ---------------
            kb_result = app.swrl_engine.infer(
                candidate_id       = cand_id,
                similarity_score   = sim_score,
                recruitment_centre = centre,
                timestamp          = datetime.utcnow()
            )

            # ---- Step 4: DS evidence fusion -----------------
            dl_mass = DempsterShaferFusion.similarity_to_mass(sim_score)
            kb_mass = kb_result['belief_mass']

            # Compute conflict K for audit trace
            conflict_k = app.ds_fusion._compute_conflict(dl_mass, kb_mass)

            belief = app.ds_fusion.combine(
                dl_mass         = dl_mass,
                kb_mass         = kb_mass,
                liveness_belief = liveness
            )

            # ---- Step 5: Threshold decision -----------------
            decision    = 1 if belief > cfg.DS_FUSION_THRESHOLD else 0
            latency_ms  = _now_ms() - start_ms

            # ---- Step 6: Audit log (PDPA compliant) ---------
            log = _save_log(
                app,
                candidate_id     = cand_id,
                similarity_score = sim_score,
                liveness_score   = liveness,
                final_belief     = belief,
                decision         = decision,
                conflict_k       = conflict_k,
                fraud_flags      = kb_result['flags'],
                swrl_rules_fired = kb_result['rules_fired'],
                severity         = kb_result['severity'],
                centre           = centre,
                device_id        = device_id,
                latency_ms       = latency_ms,
                user_id          = get_jwt_identity()
            )

            logger.info(
                "Verify: cand=%s decision=%d belief=%.3f "
                "sim=%.3f liveness=%.3f k=%.3f latency=%dms",
                cand_id, decision, belief, sim_score,
                liveness, conflict_k, latency_ms
            )

            return jsonify({
                'decision':         decision,
                'belief':           round(belief, 4),
                'confidence':       round(abs(belief - 0.5) * 2, 4),
                'similarity_score': round(sim_score, 4),
                'liveness_score':   round(liveness, 4),
                'conflict_k':       round(conflict_k, 4),
                'fraud_flags':      kb_result['flags'],
                'swrl_rules_fired': kb_result['rules_fired'],
                'severity':         kb_result['severity'],
                'timestamp':        datetime.utcnow().isoformat(),
                'audit_log_id':     log.id
            }), 200

        except Exception as exc:
            logger.error("Verify error: %s", str(exc), exc_info=True)
            return jsonify({'error': str(exc)}), 500

    # ----------------------------------------------------------
    # VERIFY — Batch
    # ----------------------------------------------------------

    @app.route('/api/v1/verify/batch', methods=['POST'])
    @jwt_required()
    def verify_batch():
        """
        Batch verification for high-throughput recruitment scenarios.

        Request JSON:
            candidates : list[dict]  — each with same fields as /verify

        Response JSON:
            results : list[dict]  — one result per candidate
        """
        data       = request.get_json() or {}
        candidates = data.get('candidates', [])

        if not isinstance(candidates, list):
            return jsonify({'error': "'candidates' must be a list"}), 400

        results = [
            _verify_single(app, candidate, get_jwt_identity())
            for candidate in candidates
        ]

        logger.info(
            "Batch verify: %d candidates processed", len(results)
        )
        return jsonify({'results': results, 'count': len(results)}), 200

    # ----------------------------------------------------------
    # ENROL — New candidate
    # ----------------------------------------------------------

    @app.route('/api/v1/enrol', methods=['POST'])
    @jwt_required()
    @require_role('admin', 'operator')
    def enrol():
        """
        Enrol a new candidate fingerprint template.
        Only hashed embedding is stored (PDPA compliance).

        Request JSON:
            candidate_id : str
            embedding    : list[float]  512-D vector
            sensor_model : str          (optional)
            centre       : str          (optional)
        """
        data         = request.get_json() or {}
        candidate_id = data.get('candidate_id', '').strip()
        embedding    = data.get('embedding')
        sensor_model = data.get('sensor_model', 'R307')
        centre       = data.get('centre', 'Unknown')

        if not candidate_id:
            return jsonify({'error': 'candidate_id is required'}), 400
        if embedding is None:
            return jsonify({'error': 'embedding is required'}), 400
        if len(embedding) != cfg.FAISS_EMBEDDING_DIM:
            return jsonify({
                'error': f'embedding must have {cfg.FAISS_EMBEDDING_DIM} dimensions'
            }), 400

        # Hash the embedding — raw biometric data never stored
        emb_bytes    = np.array(embedding, dtype=np.float32).tobytes()
        emb_hash     = hashlib.sha256(emb_bytes).hexdigest()

        template = BiometricTemplate(
            candidate_id      = candidate_id,
            embedding_hash    = emb_hash,
            sensor_model      = sensor_model,
            enrolment_centre  = centre
        )
        db.session.add(template)
        db.session.commit()

        logger.info(
            "Enrolment: cand=%s sensor=%s centre=%s",
            candidate_id, sensor_model, centre
        )
        return jsonify({
            'message':      'Candidate enrolled successfully',
            'template_id':  template.id,
            'candidate_id': candidate_id,
            'timestamp':    datetime.utcnow().isoformat()
        }), 201

    # ----------------------------------------------------------
    # AUDIT LOGS
    # ----------------------------------------------------------

    @app.route('/api/v1/audit-logs', methods=['GET'])
    @jwt_required()
    @require_role('admin', 'auditor')
    def audit_logs():
        """
        Retrieve paginated, immutable audit logs.
        Accessible to admin and auditor roles only (PDPA compliance).

        Query params:
            page     : int  (default 1)
            per_page : int  (default 50, max 200)
        """
        page     = request.args.get('page',     1,   type=int)
        per_page = min(
            request.args.get('per_page', 50, type=int),
            cfg.MAX_PAGE_SIZE
        )

        pagination = (
            VerificationLog.query
            .order_by(VerificationLog.timestamp.desc())
            .paginate(page=page, per_page=per_page, error_out=False)
        )

        return jsonify({
            'total':        pagination.total,
            'pages':        pagination.pages,
            'current_page': page,
            'per_page':     per_page,
            'logs':         [log.to_dict() for log in pagination.items]
        }), 200

    # ----------------------------------------------------------
    # DASHBOARD METRICS
    # ----------------------------------------------------------

    @app.route('/api/v1/dashboard/metrics', methods=['GET'])
    @jwt_required()
    def dashboard_metrics():
        """
        Real-time verification metrics for the React dashboard.

        Query params:
            hours : int  (default 24) — look-back window
        """
        hours      = request.args.get('hours', 24, type=int)
        since      = datetime.utcnow() - timedelta(hours=hours)

        logs = (
            VerificationLog.query
            .filter(VerificationLog.timestamp >= since)
            .all()
        )

        if not logs:
            return jsonify({
                'error': 'No verification data in the specified time window'
            }), 404

        total         = len(logs)
        accepted      = sum(1 for log in logs if log.decision == 1)
        rejected      = total - accepted
        avg_belief    = (
            sum(log.final_belief or 0.0 for log in logs) / total
        )
        avg_latency   = (
            sum(log.total_latency_ms or 0.0 for log in logs) / total
        )
        spoof_count   = sum(
            1 for log in logs
            if log.fraud_flags and 'liveness_failed' in log.fraud_flags
        )
        high_severity = sum(
            1 for log in logs if log.severity == 'high'
        )

        return jsonify({
            'time_window_hours':    hours,
            'total_verifications':  total,
            'successful_matches':   accepted,
            'rejected_matches':     rejected,
            'success_rate':         round(accepted / total, 4),
            'average_belief_score': round(avg_belief, 4),
            'average_latency_ms':   round(avg_latency, 2),
            'spoof_rejections':     spoof_count,
            'high_severity_events': high_severity,
            'timestamp':            datetime.utcnow().isoformat()
        }), 200

    # ----------------------------------------------------------
    # CANDIDATE HISTORY
    # ----------------------------------------------------------

    @app.route('/api/v1/candidate/<string:candidate_id>/history',
               methods=['GET'])
    @jwt_required()
    @require_role('admin', 'auditor')
    def candidate_history(candidate_id):
        """
        Retrieve verification history for a specific candidate.
        Admin / auditor access only.
        """
        limit = min(request.args.get('limit', 20, type=int), 100)

        logs = (
            VerificationLog.query
            .filter_by(candidate_id=candidate_id)
            .order_by(VerificationLog.timestamp.desc())
            .limit(limit)
            .all()
        )

        return jsonify({
            'candidate_id': candidate_id,
            'count':        len(logs),
            'history':      [log.to_dict() for log in logs]
        }), 200

    # ----------------------------------------------------------
    # Error Handlers
    # ----------------------------------------------------------

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify({'error': 'Method not allowed'}), 405

    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Internal error: %s", str(error))
        return jsonify({'error': 'Internal server error'}), 500

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': f'Bad request: {str(error)}'}), 400


# ============================================================
# Private Helper Functions
# ============================================================

def _query_nia_gallery(app: Flask,
                        embedding: list,
                        candidate_id: str):
    """
    Query the NIA identity gallery using the NIAGateway client.
    Returns (similarity_score, gallery_id).

    Falls back to a mock score when gateway is unreachable
    (development mode only).
    """
    country = _infer_country(candidate_id)

    result = app.nia_gateway.query_identity(
        candidate_id = candidate_id,
        embedding    = embedding,
        country      = country
    )

    if 'error' in result and app.hkb_config.__class__.__name__ != 'ProductionConfig':
        # Development fallback — deterministic mock score
        logger.warning(
            "NIA gateway unreachable — using mock score (dev mode)"
        )
        mock_score = float(
            np.clip(
                np.dot(embedding[:10], embedding[1:11]) /
                (np.linalg.norm(embedding[:10]) * np.linalg.norm(embedding[1:11]) + 1e-8),
                0.0, 1.0
            )
        )
        return mock_score, f'MOCK_GALLERY_{candidate_id}'

    return (
        float(result.get('match_score', 0.0)),
        result.get('gallery_id', '')
    )


def _verify_single(app: Flask,
                    candidate: dict,
                    user_id: int) -> dict:
    """Process a single candidate dict (used by batch endpoint)."""
    embedding = candidate.get('fingerprint_embedding', [])
    liveness  = _safe_float(candidate.get('liveness_score', 0.0))
    cand_id   = candidate.get('candidate_id', 'UNKNOWN')
    centre    = candidate.get('recruitment_centre', 'Unknown')
    cfg       = app.hkb_config

    if liveness < cfg.LIVENESS_THRESHOLD:
        return {
            'candidate_id': cand_id,
            'decision':     0,
            'belief':       0.0,
            'reason':       'Liveness failed'
        }

    try:
        sim_score, _ = _query_nia_gallery(app, embedding, cand_id)

        kb_result = app.swrl_engine.infer(
            candidate_id       = cand_id,
            similarity_score   = sim_score,
            recruitment_centre = centre,
            timestamp          = datetime.utcnow()
        )

        dl_mass = DempsterShaferFusion.similarity_to_mass(sim_score)
        belief  = app.ds_fusion.combine(
            dl_mass         = dl_mass,
            kb_mass         = kb_result['belief_mass'],
            liveness_belief = liveness
        )
        decision = 1 if belief > cfg.DS_FUSION_THRESHOLD else 0

        return {
            'candidate_id': cand_id,
            'decision':     decision,
            'belief':       round(belief, 4),
            'fraud_flags':  kb_result['flags']
        }

    except Exception as exc:
        logger.error(
            "Batch verify error for %s: %s", cand_id, str(exc)
        )
        return {
            'candidate_id': cand_id,
            'decision':     0,
            'belief':       0.0,
            'error':        str(exc)
        }


def _save_log(app: Flask,
              candidate_id:     str,
              similarity_score: float,
              liveness_score:   float,
              final_belief:     float,
              decision:         int,
              conflict_k:       float,
              fraud_flags:      list,
              swrl_rules_fired: list,
              severity:         str,
              centre:           str,
              device_id:        str,
              latency_ms:       float,
              user_id:          int) -> VerificationLog:
    """Create and persist an immutable VerificationLog entry."""
    log = VerificationLog(
        candidate_id        = candidate_id,
        similarity_score    = round(similarity_score, 6),
        liveness_score      = round(liveness_score,   6),
        final_belief        = round(final_belief,      6),
        decision            = decision,
        decision_threshold  = app.hkb_config.DS_FUSION_THRESHOLD,
        conflict_k          = round(conflict_k, 6) if conflict_k is not None else None,
        fraud_flags         = json.dumps(fraud_flags),
        swrl_rules_fired    = json.dumps(swrl_rules_fired),
        severity            = severity,
        recruitment_centre  = centre,
        device_id           = device_id,
        total_latency_ms    = round(latency_ms, 2),
        timestamp           = datetime.utcnow(),
        user_id             = user_id
    )
    db.session.add(log)
    db.session.commit()
    return log


def _db_ping(app: Flask) -> str:
    """Return 'connected' if database is reachable, else 'disconnected'."""
    try:
        with app.app_context():
            db.session.execute(db.text('SELECT 1'))
        return 'connected'
    except Exception:
        return 'disconnected'


def _now_ms() -> int:
    """Return current time in integer milliseconds."""
    return int(datetime.utcnow().timestamp() * 1000)


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_country(candidate_id: str) -> str:
    """
    Infer the NIA country from the candidate ID prefix.
    Convention: TZ-xxx → Tanzania, KE-xxx → Kenya, UG-xxx → Uganda.
    Defaults to Tanzania (PSRS context).
    """
    prefix = candidate_id[:2].upper()
    return {'KE': 'KE', 'UG': 'UG'}.get(prefix, 'TZ')


# ============================================================
# CLI Entry Point
# ============================================================

# Module-level app instance (for pytest and direct invocation)
app = create_app()


if __name__ == '__main__':
    port  = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'

    logger.info("Starting HKB-BV API on port %d (debug=%s)", port, debug)

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
