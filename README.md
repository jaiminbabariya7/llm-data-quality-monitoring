# LLM-Powered Data Quality Monitoring

An LLM-powered data quality monitoring system that profiles **BigQuery** tables, detects **statistical anomalies**, and uses **GPT-4** to generate plain-English explanations of the issues it finds - served through a **FastAPI** API and orchestrated by **Apache Airflow**.

> Built on synthetic / sample data for portfolio demonstration. No real company or customer data.

## Architecture

```
BigQuery tables
      |
      v
data_profiler.py      column stats: nulls, uniqueness, min/max, distributions, freshness
      |
      v
anomaly_detector.py   statistical anomalies (z-score / IQR / drift vs. rolling baseline)
      |
      v
llm_explainer.py      GPT-4 -> plain-English "what changed and why it matters"
      |
      +--> api/main.py        FastAPI service (profile / anomalies / explain / health)
      |
      +--> alert_manager.py   quality alerts + human-readable summaries
      |
      v
dags/quality_monitor_dag.py   Airflow DAG: scheduled daily quality run + alerting
```

## Components

| Module | Responsibility |
| --- | --- |
| `src/bigquery_connector.py` | Connects to BigQuery and pulls table data / metadata |
| `src/data_profiler.py` | Profiles tables - null %, uniqueness, ranges, distributions, freshness |
| `src/anomaly_detector.py` | Flags statistical anomalies against a rolling baseline |
| `src/llm_explainer.py` | Sends anomalies to GPT-4 for concise, business-readable explanations |
| `src/alert_manager.py` | Formats and dispatches data-quality alerts |
| `api/main.py`, `api/schemas.py` | FastAPI service + request/response models |
| `dags/quality_monitor_dag.py` | Airflow DAG that schedules the daily monitoring run |
| `tests/` | Unit tests for the profiler and anomaly detector |

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 |
| Warehouse | Google BigQuery |
| LLM | OpenAI GPT-4 |
| API | FastAPI |
| Orchestration | Apache Airflow |
| Packaging | Docker + docker-compose |
| Practices | MLOps, data quality, anomaly detection |

## Getting Started

```bash
# 1. Clone
git clone https://github.com/jaiminbabariya7/llm-data-quality-monitoring.git
cd llm-data-quality-monitoring

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Fill in GCP_PROJECT_ID, BigQuery dataset, and OPENAI_API_KEY

# 4. Run the API
uvicorn api.main:app --reload

# 5. Or run the full stack with Docker
cd docker && docker-compose up --build

# 6. Run tests
pytest tests/ -v
```

## Skills Demonstrated

Python | Google BigQuery | Data Profiling | Statistical Anomaly Detection | OpenAI GPT-4 / LLM | FastAPI | Apache Airflow | Docker | MLOps | Data Quality
