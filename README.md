# 🌌 Project Aether: Continuous-Time Policy Shock Simulator

**A Liquid Neural Controlled Differential Equation (CDE) framework for adapting to irregular healthcare claim timelines and sudden regulatory policy shifts.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 🚀 Overview

Standard discrete-time AI models (LSTMs, Transformers) struggle with two fundamental challenges in healthcare administration:
1. **Irregular Temporal Spacing:** Patient claims arrive at unpredictable intervals (e.g., Day 1, Day 14, Day 45). Discrete models force this irregular pacing into rigid, padded time-steps, losing critical temporal context.
2. **Concept Drift:** Third-Party Administrators (TPAs) frequently update denial rules. When a policy changes, static models become instantly obsolete and require weeks of expensive retraining.

**Project Aether** addresses both limitations using a **Liquid Neural Controlled Differential Equation (CDE)** framework. Instead of treating time as rigid, discrete steps, it maps irregular claims into a continuous mathematical path $X(t)$ using Hermite Cubic Splines. When a policy shock occurs, the model adapts in real time by mathematically integrating the new derivative, smoothly adjusting its latent trajectory without requiring a complete retraining cycle.

---

## 🧠 Core Mathematical Innovations

* **Neural CDEs (`torchcde`):** Maps irregular, asynchronous claim events into a continuous latent space, natively handling the variable intervals between patient visits without requiring artificial data imputation.
* **Liquid Time-Constants (LTC):** The vector field $F(t, h)$ dynamically adapts its internal time-scales ($\tau$) based on incoming features, enabling robust handling of sudden distribution shifts.
* **Continuous Policy Shocks:** Simulates regulatory or TPA rule changes by mutating the continuous input path $X(t)$ rather than merely perturbing network weights, providing a mathematically rigorous stress test of the model's continuous dynamics.
* **Saturation-Resistant Activations:** Replaces standard `tanh` activations with `softsign` functions within the LTC cell to prevent gradient saturation during massive distribution shifts, ensuring the model remains highly sensitive to incoming data variations.

---

## 🌪️ The Hero Feature: Policy Shock Simulator

The core innovation of Project Aether is the **Shock Test Module**. It allows operators to simulate distinct regulatory shocks and observe how the continuous trajectory adapts dynamically:

* **Step Shock:** A permanent, structural shift in reimbursement rules.
* **Impulse Shock:** A transient anomaly (e.g., a localized public health spike or short-term system outage).
* **Ramp Shock:** A gradual phase-out or transition period for specific procedure codes.

### Rigorous Evaluation Metrics
Rather than relying on static performance baselines or conventional accuracy scores that fail to capture continuous-time variations, the simulator quantifies **dynamic contextual adaptation** using three mathematically rigorous metrics:

1. **Latent Shift Magnitude:** The total geometric distance traversed by the latent state immediately following the shock, validating that the system registered the environmental change.
2. **Final State Deviation:** The distance between the post-shock steady state and the original baseline, measuring the model's capacity to quantify the permanent impact of the new regulatory regime.
3. **Settling Time:** The duration required for the trajectory to stabilize and lock into the new systemic operational baseline, evaluating the speed of adaptation.

---

## 🏗️ System Architecture & Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **ML Engine** | PyTorch, `torchcde` | Liquid Neural CDE implementation, LTC Cell, and ODE integration (`rk4` method). |
| **Backend API** | FastAPI, Uvicorn | High-performance inference, spline generation, and shock simulation endpoints. |
| **Frontend UI** | Next.js 14, React, TypeScript | Interactive dashboard, real-time trajectory visualizer, and shock controls. |
| **Visualization** | Recharts, Framer Motion | Low-latency telemetry interface, latent state plotting, and gap markers. |
| **Data Pipeline** | Pandas, Scikit-Learn | CMS data parsing, Robust Scaling, and Hermite cubic coefficient generation. |

---

## 📂 Project Structure

```text
project-aether/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── math_engine.py      # Liquid CDE, LTC Cell, and Shock Simulator logic
│   │   │   └── config.py           # Pydantic settings and environment configs
│   │   └── api/
│   │       └── routes.py           # FastAPI endpoints (Predict, Shock-Test, Telemetry)
│   ├── data_pipeline/
│   │   ├── cde_formatter.py        # Converts claim events to Hermite spline coefficients
│   │   └── cms_loader.py           # Synthetic CMS dataset generator
│   └── models/
│       └── aether_model.pt         # Trained Liquid CDE weights
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # Main Command Dashboard
│   │   ├── visualizer/page.tsx     # Continuous Latent Trajectory Oscilloscope
│   │   ├── shock-test/page.tsx     # Policy Shock Simulator UI
│   │   └── logs/page.tsx           # Raw Telemetry Logs Table
│   ├── components/                 # Sidebar, MainGrid, UI components
│   └── lib/api/aether-client.ts    # TypeScript API contracts and fetch client
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Backend (Python FastAPI)

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --port 8000
```
*The API will be available at `http://localhost:8000`. Interactive OpenAPI documentation can be accessed at `http://localhost:8000/docs`.*

### 2. Frontend (Next.js)

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
*The user interface will be available at `http://localhost:3000`.*

---

## 📡 API Endpoints

The backend exposes the following core endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/predict` | Executes Liquid CDE inference on a sequence of irregular claim events. |
| `POST` | `/api/v1/shock-test` | Simulates a TPA policy shock and returns adaptation metrics/trajectories. |
| `GET` | `/api/v1/sample-data` | Fetches a pre-generated synthetic sample sequence for UI demonstrations. |
| `GET` | `/api/v1/telemetry/dashboard` | Compiles system health metrics, ingestion loads, and drift alerts. |
| `GET` | `/api/v1/telemetry/trajectory` | Stream continuous trajectory arrays for the visualizer oscilloscope. |
| `GET` | `/api/v1/telemetry/logs` | Retrieves dense, raw telemetry rows for the tabular log viewer. |

---

## 🎥 Demo & Walkthrough

> **[📺 Watch the Video Demo on LinkedIn/YouTube]** *(Insert your video link here)*

**Key Highlights to Observe in the Demonstration:**
1. Navigate to the **Policy Shock Simulator** interface.
2. Configure the Shock Type to **"Step"** and adjust the **Shock Strength** to 100%.
3. Observe how the **Shocked Trajectory (Green Line)** smoothly modulates away from the **Baseline Trajectory (Purple Line)**.
4. Note the dynamic adjustment of the **Latent Shift Magnitude**, verifying that the system contextually absorbs the policy shift mathematically without destabilizing.

---

## 🏭 Production Readiness & Future Roadmap

> **Note on Dataset Architecture:** This repository utilizes a **Synthetic Digital Twin Environment** (a mathematically rigorous 600-row simulated CMS dataset) to generate irregular claim sequences and stress-test the model. This architectural choice enables full validation of the continuous-time mechanics while maintaining absolute compliance with data privacy principles and eliminating dependencies on siloed, HIPAA-restricted healthcare records.

To transition Project Aether from a research-grade framework to an enterprise-tier production system, the following infrastructure additions are planned:
* **HL7/FHIR Ingestion:** Integration with standard healthcare data exchange protocols to ingest live clinical and administrative transactions directly from EHR systems.
* **Distributed Stream Processing:** Transitioning from polling mechanisms to an event-driven architecture utilizing Apache Kafka or AWS Kinesis for real-time asynchronous claim routing.
* **Time-Series Engine:** Upgrading the storage layer from transient in-memory structures to optimized time-series databases like TimescaleDB or InfluxDB.
* **Enterprise MLOps:** Deploying MLflow or Weights & Biases pipelines to manage model registry variants, track continuous evaluations, and flag real-world latent drift.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

**Built with ❤️ and a lot of calculus.** *If you find this research or implementation useful, please consider giving the repository a ⭐!*