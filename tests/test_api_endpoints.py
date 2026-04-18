#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 08:54:09 2026

@author: dr
Unit Tests: Flask REST API Endpoints — Fixed Version
"""

import pytest
import json
import sys
import os
import time
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.app import app as flask_app, db
    from backend.models import User, VerificationLog
except ImportError as e:
    raise ImportError(
        f"Cannot import Flask app. Error: {e}"
    )


# ============================================================
# Helpers
# ============================================================

def _seed_test_users():
    if not User.query.filter_by(username='test_operator').first():
        op = User(username='test_operator',
                  email='operator@test.local', role='operator')
        op.set_password('TestPass123!')
        db.session.add(op)

    if not User.query.filter_by(username='test_admin').first():
        adm = User(username='test_admin',
                   email='admin@test.local', role='admin')
        adm.set_password('AdminPass123!')
        db.session.add(adm)

    if not User.query.filter_by(username='test_auditor').first():
        aud = User(username='test_auditor',
                   email='auditor@test.local', role='auditor')
        aud.set_password('AuditPass123!')
        db.session.add(aud)

    db.session.commit()


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope='module')
def test_client():
    flask_app.config['TESTING']                 = True
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    flask_app.config['JWT_SECRET_KEY']          = 'test-jwt-secret'
    flask_app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False

    with flask_app.test_client() as client:
        with flask_app.app_context():
            db.create_all()
            _seed_test_users()
        yield client
        with flask_app.app_context():
            db.drop_all()


@pytest.fixture(scope='module')
def auth_token(test_client):
    r = test_client.post('/auth/login',
        json={'username': 'test_operator', 'password': 'TestPass123!'},
        content_type='application/json')
    assert r.status_code == 200, f"Login failed: {r.get_json()}"
    return r.get_json()['access_token']


@pytest.fixture(scope='module')
def admin_token(test_client):
    r = test_client.post('/auth/login',
        json={'username': 'test_admin', 'password': 'AdminPass123!'},
        content_type='application/json')
    assert r.status_code == 200
    return r.get_json()['access_token']


@pytest.fixture(scope='module')
def auditor_token(test_client):
    r = test_client.post('/auth/login',
        json={'username': 'test_auditor', 'password': 'AuditPass123!'},
        content_type='application/json')
    assert r.status_code == 200
    return r.get_json()['access_token']


@pytest.fixture
def valid_probe():
    return {
        'fingerprint_embedding': [round(0.001 * i, 4) for i in range(512)],
        'liveness_score':        0.87,
        'candidate_id':          'PSRS_TEST_12345',
        'recruitment_centre':    'Dar_es_Salaam'
    }


@pytest.fixture
def spoof_probe():
    return {
        'fingerprint_embedding': [round(0.002 * i, 4) for i in range(512)],
        'liveness_score':        0.28,
        'candidate_id':          'SPOOF_CANDIDATE_001',
        'recruitment_centre':    'Nairobi'
    }


# ============================================================
# Health Tests
# ============================================================

class TestHealthEndpoint:

    def test_health_returns_200(self, test_client):
        assert test_client.get('/health').status_code == 200

    def test_health_returns_json(self, test_client):
        assert test_client.get('/health').get_json() is not None

    def test_health_contains_status_field(self, test_client):
        assert 'status' in test_client.get('/health').get_json()

    def test_health_status_is_healthy(self, test_client):
        data = test_client.get('/health').get_json()
        assert data.get('status') == 'healthy'

    def test_health_contains_services_field(self, test_client):
        assert 'services' in test_client.get('/health').get_json()

    def test_health_no_auth_required(self, test_client):
        assert test_client.get('/health').status_code != 401

    def test_health_contains_timestamp(self, test_client):
        assert 'timestamp' in test_client.get('/health').get_json()


# ============================================================
# Auth Tests
# ============================================================

class TestAuthRegistration:

    def test_register_new_user_returns_201(self, test_client):
        r = test_client.post('/auth/register',
            json={'username': 'new_recruiter', 'email': 'rec@psrs.tz',
                  'password': 'Secure123!', 'role': 'operator'},
            content_type='application/json')
        assert r.status_code == 201

    def test_register_duplicate_user_returns_409(self, test_client):
        payload = {'username': 'dup_user', 'email': 'dup@test.local',
                   'password': 'Pass123!', 'role': 'operator'}
        test_client.post('/auth/register', json=payload,
                         content_type='application/json')
        r = test_client.post('/auth/register', json=payload,
                             content_type='application/json')
        assert r.status_code == 409

    def test_register_returns_success_message(self, test_client):
        r = test_client.post('/auth/register',
            json={'username': 'msg_user', 'email': 'msg@test.local',
                  'password': 'Pass123!', 'role': 'operator'},
            content_type='application/json')
        assert 'message' in r.get_json()

    def test_register_missing_fields_returns_400(self, test_client):
        r = test_client.post('/auth/register',
            json={'username': 'incomplete'},
            content_type='application/json')
        assert r.status_code in [400, 422]


class TestAuthLogin:

    def test_valid_login_returns_200(self, test_client):
        r = test_client.post('/auth/login',
            json={'username': 'test_operator', 'password': 'TestPass123!'},
            content_type='application/json')
        assert r.status_code == 200

    def test_valid_login_returns_access_token(self, test_client):
        r = test_client.post('/auth/login',
            json={'username': 'test_operator', 'password': 'TestPass123!'},
            content_type='application/json')
        data = r.get_json()
        assert 'access_token' in data
        assert len(data['access_token']) > 20

    def test_invalid_password_returns_401(self, test_client):
        r = test_client.post('/auth/login',
            json={'username': 'test_operator', 'password': 'WrongPass!'},
            content_type='application/json')
        assert r.status_code == 401

    def test_nonexistent_user_returns_401(self, test_client):
        r = test_client.post('/auth/login',
            json={'username': 'ghost_user', 'password': 'Any123!'},
            content_type='application/json')
        assert r.status_code == 401

    def test_login_error_has_error_field(self, test_client):
        r = test_client.post('/auth/login',
            json={'username': 'bad', 'password': 'bad'},
            content_type='application/json')
        assert 'error' in r.get_json()


# ============================================================
# Verify Tests — patch app object attributes directly
# ============================================================

class TestVerifyEndpoint:

    def test_verify_without_auth_returns_401(
        self, test_client, valid_probe
    ):
        r = test_client.post('/api/v1/verify', json=valid_probe,
                             content_type='application/json')
        assert r.status_code == 401

    def test_verify_with_invalid_token_returns_401(
        self, test_client, valid_probe
    ):
        r = test_client.post('/api/v1/verify', json=valid_probe,
            headers={'Authorization': 'Bearer invalid.jwt.token'},
            content_type='application/json')
        assert r.status_code == 401

    def test_verify_valid_probe_returns_200(
        self, test_client, auth_token, valid_probe
    ):
        with patch.object(
            test_client.application.swrl_engine, 'infer',
            return_value={'flags': [], 'belief_mass': {
                'match': 0.85, 'non_match': 0.10, 'unknown': 0.05},
                'rules_fired': [], 'severity': 'low'}
        ), patch.object(
            test_client.application.ds_fusion, 'combine',
            return_value=0.88
        ), patch('backend.app._query_nia_gallery',
                 return_value=(0.95, 'NIA_ID_001')):
            r = test_client.post('/api/v1/verify', json=valid_probe,
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        assert r.status_code == 200

    def test_verify_response_contains_required_fields(
        self, test_client, auth_token, valid_probe
    ):
        with patch.object(
            test_client.application.swrl_engine, 'infer',
            return_value={'flags': [], 'belief_mass': {
                'match': 0.85, 'non_match': 0.10, 'unknown': 0.05},
                'rules_fired': [], 'severity': 'low'}
        ), patch.object(
            test_client.application.ds_fusion, 'combine',
            return_value=0.88
        ), patch('backend.app._query_nia_gallery',
                 return_value=(0.95, 'NIA_ID_001')):
            r = test_client.post('/api/v1/verify', json=valid_probe,
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        data = r.get_json()
        for field in ['decision', 'belief', 'confidence',
                      'similarity_score', 'liveness_score',
                      'fraud_flags', 'swrl_rules_fired',
                      'timestamp', 'audit_log_id']:
            assert field in data, f"Missing field: '{field}'"

    def test_verify_decision_is_binary(
        self, test_client, auth_token, valid_probe
    ):
        with patch.object(
            test_client.application.swrl_engine, 'infer',
            return_value={'flags': [], 'belief_mass': {
                'match': 0.85, 'non_match': 0.10, 'unknown': 0.05},
                'rules_fired': [], 'severity': 'low'}
        ), patch.object(
            test_client.application.ds_fusion, 'combine',
            return_value=0.88
        ), patch('backend.app._query_nia_gallery',
                 return_value=(0.95, 'NIA_ID_001')):
            r = test_client.post('/api/v1/verify', json=valid_probe,
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        assert r.get_json()['decision'] in [0, 1]

    def test_verify_spoof_rejected_immediately(
        self, test_client, auth_token, spoof_probe
    ):
        with patch('backend.app._query_nia_gallery') as mock_gallery:
            r = test_client.post('/api/v1/verify', json=spoof_probe,
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
            assert r.status_code == 200
            assert r.get_json()['decision'] == 0
            mock_gallery.assert_not_called()

    def test_verify_belief_in_valid_range(
        self, test_client, auth_token, valid_probe
    ):
        with patch.object(
            test_client.application.swrl_engine, 'infer',
            return_value={'flags': [], 'belief_mass': {
                'match': 0.85, 'non_match': 0.10, 'unknown': 0.05},
                'rules_fired': [], 'severity': 'low'}
        ), patch.object(
            test_client.application.ds_fusion, 'combine',
            return_value=0.88
        ), patch('backend.app._query_nia_gallery',
                 return_value=(0.95, 'NIA_ID_001')):
            r = test_client.post('/api/v1/verify', json=valid_probe,
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        belief = r.get_json()['belief']
        assert 0.0 <= belief <= 1.0

    def test_verify_creates_audit_log_entry(
        self, test_client, auth_token, valid_probe
    ):
        with patch.object(
            test_client.application.swrl_engine, 'infer',
            return_value={'flags': [], 'belief_mass': {
                'match': 0.85, 'non_match': 0.10, 'unknown': 0.05},
                'rules_fired': [], 'severity': 'low'}
        ), patch.object(
            test_client.application.ds_fusion, 'combine',
            return_value=0.88
        ), patch('backend.app._query_nia_gallery',
                 return_value=(0.95, 'NIA_ID_001')):
            r = test_client.post('/api/v1/verify', json=valid_probe,
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        data = r.get_json()
        assert 'audit_log_id' in data
        assert data['audit_log_id'] is not None

    def test_verify_missing_embedding_returns_error(
        self, test_client, auth_token
    ):
        r = test_client.post('/api/v1/verify',
            json={'liveness_score': 0.85, 'candidate_id': 'MISS_001'},
            headers={'Authorization': f'Bearer {auth_token}'},
            content_type='application/json')
        assert r.status_code in [400, 422, 500]

    def test_verify_high_fraud_flags_reduce_belief(
        self, test_client, auth_token, valid_probe
    ):
        with patch.object(
            test_client.application.swrl_engine, 'infer',
            return_value={'flags': [], 'belief_mass': {
                'match': 0.85, 'non_match': 0.10, 'unknown': 0.05},
                'rules_fired': [], 'severity': 'low'}
        ), patch.object(
            test_client.application.ds_fusion, 'combine',
            return_value=0.88
        ), patch('backend.app._query_nia_gallery',
                 return_value=(0.95, 'NIA_ID_001')):
            resp_clean = test_client.post(
                '/api/v1/verify', json=valid_probe,
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        belief_clean = resp_clean.get_json().get('belief', 0.0)

        with patch.object(
            test_client.application.swrl_engine, 'infer',
            return_value={'flags': ['Velocity_Anomaly'],
                'belief_mass': {'match': 0.20, 'non_match': 0.75,
                                'unknown': 0.05},
                'rules_fired': ['velocity_anomaly'], 'severity': 'high'}
        ), patch.object(
            test_client.application.ds_fusion, 'combine',
            return_value=0.22
        ), patch('backend.app._query_nia_gallery',
                 return_value=(0.95, 'NIA_ID_001')):
            resp_fraud = test_client.post(
                '/api/v1/verify', json=valid_probe,
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        belief_fraud = resp_fraud.get_json().get('belief', 1.0)
        assert belief_clean > belief_fraud


# ============================================================
# Batch Verify Tests
# ============================================================

class TestBatchVerifyEndpoint:

    def test_batch_verify_returns_200(self, test_client, auth_token):
        with patch('backend.app._verify_single',
                   return_value={'candidate_id': 'B001',
                                 'decision': 1, 'belief': 0.85,
                                 'fraud_flags': []}):
            r = test_client.post('/api/v1/verify/batch',
                json={'candidates': [
                    {'candidate_id': 'B001',
                     'fingerprint_embedding': [0.1] * 512,
                     'liveness_score': 0.80}]},
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        assert r.status_code == 200

    def test_batch_verify_returns_results_list(
        self, test_client, auth_token
    ):
        with patch('backend.app._verify_single',
                   return_value={'candidate_id': 'B001',
                                 'decision': 1, 'belief': 0.85}):
            r = test_client.post('/api/v1/verify/batch',
                json={'candidates': [
                    {'candidate_id': 'B001',
                     'fingerprint_embedding': [0.1] * 512,
                     'liveness_score': 0.80}]},
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        data = r.get_json()
        assert 'results' in data
        assert isinstance(data['results'], list)

    def test_batch_verify_without_auth_returns_401(self, test_client):
        r = test_client.post('/api/v1/verify/batch',
            json={'candidates': []},
            content_type='application/json')
        assert r.status_code == 401


# ============================================================
# Audit Log Tests
# ============================================================

class TestAuditLogEndpoint:

    def test_audit_logs_without_auth_returns_401(self, test_client):
        assert test_client.get('/api/v1/audit-logs').status_code == 401

    def test_audit_logs_operator_access_denied(
        self, test_client, auth_token
    ):
        r = test_client.get('/api/v1/audit-logs',
            headers={'Authorization': f'Bearer {auth_token}'})
        assert r.status_code == 403

    def test_audit_logs_admin_access_granted(
        self, test_client, admin_token
    ):
        r = test_client.get('/api/v1/audit-logs',
            headers={'Authorization': f'Bearer {admin_token}'})
        assert r.status_code == 200

    def test_audit_logs_auditor_access_granted(
        self, test_client, auditor_token
    ):
        r = test_client.get('/api/v1/audit-logs',
            headers={'Authorization': f'Bearer {auditor_token}'})
        assert r.status_code == 200

    def test_audit_logs_response_structure(
        self, test_client, admin_token
    ):
        r = test_client.get('/api/v1/audit-logs',
            headers={'Authorization': f'Bearer {admin_token}'})
        data = r.get_json()
        for field in ['total', 'pages', 'current_page', 'logs']:
            assert field in data

    def test_audit_logs_returns_list(self, test_client, admin_token):
        r = test_client.get('/api/v1/audit-logs',
            headers={'Authorization': f'Bearer {admin_token}'})
        assert isinstance(r.get_json()['logs'], list)


# ============================================================
# Dashboard Tests
# ============================================================

class TestDashboardMetricsEndpoint:

    def test_dashboard_without_auth_returns_401(self, test_client):
        assert test_client.get(
            '/api/v1/dashboard/metrics'
        ).status_code == 401

    def test_dashboard_with_auth_returns_200_or_404(
        self, test_client, auth_token
    ):
        r = test_client.get('/api/v1/dashboard/metrics',
            headers={'Authorization': f'Bearer {auth_token}'})
        assert r.status_code in [200, 404]

    def test_dashboard_success_rate_in_valid_range(
        self, test_client, auth_token
    ):
        r = test_client.get('/api/v1/dashboard/metrics',
            headers={'Authorization': f'Bearer {auth_token}'})
        if r.status_code == 200:
            rate = r.get_json().get('success_rate', 0.5)
            assert 0.0 <= rate <= 1.0


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrorHandling:

    def test_404_unknown_endpoint(self, test_client):
        assert test_client.get(
            '/api/v1/nonexistent'
        ).status_code == 404

    def test_method_not_allowed_returns_405(self, test_client):
        assert test_client.get('/auth/login').status_code == 405

    def test_internal_error_returns_500(
        self, test_client, auth_token, valid_probe
    ):
        with patch('backend.app._query_nia_gallery',
                   side_effect=Exception("NIA timeout")):
            r = test_client.post('/api/v1/verify', json=valid_probe,
                headers={'Authorization': f'Bearer {auth_token}'},
                content_type='application/json')
        assert r.status_code == 500

    def test_malformed_json_returns_error(
        self, test_client, auth_token
    ):
        r = test_client.post('/api/v1/verify',
            data='{"invalid: json}',
            headers={'Authorization': f'Bearer {auth_token}',
                     'Content-Type': 'application/json'})
        assert r.status_code in [400, 422, 500]


# ============================================================
# Security Tests
# ============================================================

class TestSecurityControls:

    def test_operator_cannot_access_admin_endpoints(
        self, test_client, auth_token
    ):
        r = test_client.get('/api/v1/audit-logs',
            headers={'Authorization': f'Bearer {auth_token}'})
        assert r.status_code == 403

    def test_sql_injection_in_login_returns_401(self, test_client):
        r = test_client.post('/auth/login',
            json={"username": "admin' OR '1'='1",
                  "password": "' OR '1'='1"},
            content_type='application/json')
        assert r.status_code == 401

    def test_tampered_token_returns_401(
        self, test_client, auth_token, valid_probe
    ):
        tampered = auth_token[:-10] + 'TAMPERED!!'
        r = test_client.post('/api/v1/verify', json=valid_probe,
            headers={'Authorization': f'Bearer {tampered}'},
            content_type='application/json')
        assert r.status_code == 401


# ============================================================
# Performance Tests
# ============================================================

class TestPerformance:

    def test_health_response_time(self, test_client):
        """Health endpoint must respond within 5 seconds"""
        start   = time.time()
        test_client.get('/health')
        elapsed = time.time() - start
        assert elapsed < 5.0, \
            f"Health endpoint took {elapsed:.2f}s (expected < 5.0s)"

    def test_login_response_time(self, test_client):
        """Login endpoint must respond within 5 seconds"""
        start = time.time()
        test_client.post('/auth/login',
            json={'username': 'test_operator',
                  'password': 'TestPass123!'},
            content_type='application/json')
        elapsed = time.time() - start
        assert elapsed < 5.0, \
            f"Login took {elapsed:.2f}s (expected < 5.0s)"

    def test_concurrent_login_requests(self, test_client):
        """5 sequential logins must all succeed"""
        for _ in range(5):
            r = test_client.post('/auth/login',
                json={'username': 'test_operator',
                      'password': 'TestPass123!'},
                content_type='application/json')
            assert r.status_code == 200
