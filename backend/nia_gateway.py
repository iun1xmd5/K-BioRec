"""
HKB-BV NIA Gateway Client
Interfaces with NIDA (Tanzania), NIIMS/Huduma Namba (Kenya),
and NIRA (Uganda) national identity databases
"""

import logging
import requests
from typing import Tuple, Optional, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class NIAGateway:
    """
    Unified client for querying East African national identity
    authority (NIA) databases via mTLS-secured REST APIs.
    """

    def __init__(self,
                 nida_api_url:  str = '',
                 niims_api_url: str = '',
                 nira_api_url:  str = '',
                 timeout:       int = 10,
                 max_retries:   int = 3,
                 ca_cert:       str = None,
                 client_cert:   Tuple[str, str] = None):
        """
        Args:
            nida_api_url:  NIDA REST API base URL (Tanzania)
            niims_api_url: NIIMS REST API base URL (Kenya)
            nira_api_url:  NIRA REST API base URL (Uganda)
            timeout:       Request timeout in seconds
            max_retries:   Maximum retry attempts on failure
            ca_cert:       Path to CA certificate for TLS verification
            client_cert:   Tuple of (cert_path, key_path) for mTLS
        """
        self.endpoints = {
            'NIDA':  nida_api_url,
            'NIIMS': niims_api_url,
            'NIRA':  nira_api_url
        }
        self.timeout     = timeout
        self.ca_cert     = ca_cert
        self.client_cert = client_cert

        # Configure session with retry logic
        self.session = self._build_session(max_retries)

        logger.info(
            f"NIAGateway initialised: NIDA={nida_api_url}, "
            f"NIIMS={niims_api_url}, NIRA={nira_api_url}"
        )

    # ============================================================
    # Public Methods
    # ============================================================

    def query_identity(self,
                       candidate_id: str,
                       embedding: list,
                       country: str = 'TZ') -> Dict:
        """
        Query the appropriate NIA database for a candidate.

        Args:
            candidate_id: National ID or recruitment reference number
            embedding:    512-D fingerprint embedding vector
            country:      ISO country code ('TZ', 'KE', 'UG')

        Returns:
            {
                'found':       bool,
                'match_score': float,
                'gallery_id':  str,
                'metadata':    dict
            }
        """
        gateway = self._select_gateway(country)

        if not gateway:
            logger.warning(f"No gateway configured for country '{country}'")
            return self._not_found_response()

        try:
            result = self._query_gateway(gateway, candidate_id, embedding)
            logger.info(
                f"NIA query: candidate={candidate_id}, "
                f"country={country}, found={result['found']}"
            )
            return result

        except requests.exceptions.Timeout:
            logger.error(f"NIA timeout: gateway={gateway}, candidate={candidate_id}")
            return self._error_response('Gateway timeout')

        except requests.exceptions.ConnectionError:
            logger.error(f"NIA connection error: gateway={gateway}")
            return self._error_response('Connection error')

        except Exception as e:
            logger.error(f"NIA unexpected error: {str(e)}")
            return self._error_response(str(e))

    def is_reachable(self, country: str = 'TZ') -> bool:
        """Check if NIA gateway is reachable."""
        gateway = self._select_gateway(country)
        if not gateway:
            return False

        try:
            response = self.session.get(
                f"{gateway}/health",
                timeout=5
            )
            return response.status_code == 200

        except Exception:
            return False

    def enrol_candidate(self,
                        candidate_id: str,
                        embedding: list,
                        metadata: dict,
                        country: str = 'TZ') -> Dict:
        """
        Enrol a new candidate in the NIA gallery.

        Args:
            candidate_id: Candidate identifier
            embedding:    512-D fingerprint embedding
            metadata:     Candidate metadata (name, DOB, etc.)
            country:      ISO country code

        Returns:
            {'success': bool, 'gallery_id': str, 'message': str}
        """
        gateway = self._select_gateway(country)

        if not gateway:
            return {'success': False, 'message': 'No gateway for country'}

        try:
            payload = {
                'candidate_id': candidate_id,
                'embedding':    embedding,
                'metadata':     metadata
            }
            response = self.session.post(
                f"{gateway}/enrol",
                json=payload,
                timeout=self.timeout,
                verify=self.ca_cert or True,
                cert=self.client_cert
            )

            if response.status_code == 201:
                data = response.json()
                return {
                    'success':    True,
                    'gallery_id': data.get('gallery_id', ''),
                    'message':    'Enrolment successful'
                }

            return {
                'success': False,
                'message': f'Enrolment failed: HTTP {response.status_code}'
            }

        except Exception as e:
            return {'success': False, 'message': str(e)}

    # ============================================================
    # Private Helpers
    # ============================================================

    def _build_session(self, max_retries: int) -> requests.Session:
        """Build requests session with retry logic and TLS."""
        session = requests.Session()

        retry = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        session.mount('http://',  adapter)

        # Set default headers
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept':       'application/json',
            'User-Agent':   'HKB-BV/1.0'
        })

        return session

    def _select_gateway(self, country: str) -> Optional[str]:
        """Map ISO country code to NIA gateway URL."""
        mapping = {
            'TZ': self.endpoints.get('NIDA'),
            'KE': self.endpoints.get('NIIMS'),
            'UG': self.endpoints.get('NIRA')
        }
        return mapping.get(country.upper())

    def _query_gateway(self,
                       gateway_url: str,
                       candidate_id: str,
                       embedding: list) -> Dict:
        """Execute 1:N query against a specific NIA gateway."""
        payload = {
            'candidate_id': candidate_id,
            'embedding':    embedding[:512],  # Enforce 512-D
            'top_k':        1
        }

        response = self.session.post(
            f"{gateway_url}/verify",
            json=payload,
            timeout=self.timeout,
            verify=self.ca_cert or True,
            cert=self.client_cert
        )

        if response.status_code == 200:
            data = response.json()
            return {
                'found':       data.get('found', False),
                'match_score': float(data.get('match_score', 0.0)),
                'gallery_id':  data.get('gallery_id', ''),
                'metadata':    data.get('metadata', {})
            }

        if response.status_code == 404:
            return self._not_found_response()

        logger.warning(
            f"NIA gateway returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )
        return self._error_response(f'HTTP {response.status_code}')

    def _not_found_response(self) -> Dict:
        return {
            'found':       False,
            'match_score': 0.0,
            'gallery_id':  '',
            'metadata':    {}
        }

    def _error_response(self, message: str) -> Dict:
        return {
            'found':       False,
            'match_score': 0.0,
            'gallery_id':  '',
            'metadata':    {},
            'error':       message
        }
