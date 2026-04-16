# HKB-BV: Hybrid Knowledge-Based Biometric Verification Framework

A low-cost, deployable biometric verification system for East African public-sector recruitment using IoT edge sensing, deep learning, SWRL ontology reasoning, and Dempster–Shafer evidence fusion.

## Features
- **ESP32 Edge Device**: Low-cost fingerprint acquisition with real-time liveness detection - **Fuzzy Liveness Detection**: Complementary pore density and ridge quality fusion (α=0.4, SRR 98.5%)
- **SWRL Ontology Reasoning**: Contextual fraud inference (velocity anomalies, geospatial conflicts)
- **Dempster–Shafer Fusion**: Uncertainty-aware evidence combination (AUC 0.968, EER 1.27%)
- **National-Scale Scalability**: FAISS indexing supports 10M identities at 28ms latency
- **Explainable Decisions**: SHAP audit traces for regulatory compliance (PDPA, DPA, DPPA)

## Performance

| Metric | Value | Dataset |
|--------|-------|---------|
| AUC | 0.968 ± 0.002 | Synthetic PSRS |
| EER | 1.27% ± 0.06 | Mean (3 datasets) |
| SRR | 98% ± 1.0 | Spoof rejection |
| Latency | 1.91s | End-to-end (1.72s stable, 2.10s intermittent) |
| Throughput | 35 verifications/min | Stable 4G |
| Scalability | 28ms @ 10M identities | FAISS indexing |

## Quick Start

### Prerequisites
- Python 3.8+
- Arduino IDE 1.8.13+ (ESP32 board support)
- Docker & Docker Compose (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/kbiodet.git
cd kbiodet

# Install backend dependencies
pip install -r requirements.txt

# Install edge firmware dependencies (Arduino IDE)
# Install board: esp32:esp32:esp32wroom32

