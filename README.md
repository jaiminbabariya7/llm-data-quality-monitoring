# LLM-Powered Data Quality Monitoring

An LLM-powered data quality monitoring system that profiles **BigQuery** tables, detects **statistical anomalies**, and uses an **OpenAI** model (`gpt-4o-mini` by default, configurable via `LLM_MODEL`) to generate plain-English explanations of the issues it finds - served through a **FastAPI** API and orchestrated by **Apache Airflow**.

> Built on synthetic / sample data for portfolio demonstration. No real company or customer data.

## Architecture

```
BigQuery tables
      |
      v
data_profiler.py      column stats: nulls, uniqueness, min/max, distributions, freshness
      |
      v
anomaly_detector.py   statistical anomalies (null rates / z-score + IQR outliers / mean drift / volume drift / schema changes)
      |
      v
llm_explainer.py      OpenAI (rule-based fallback) -> plain-English "what changed and why it matters"
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
| `src/data_profiler.py` | Profiles tables - null %, uniqueness, ranges, distributions, freshness; saves each profile as the baseline for the next run |
| `src/anomaly_detector.py` | Runs the checks below against the previous profile and the 7-day row-count history |
| `src/llm_explainer.py` | Sends anomalies to an OpenAI model for concise, business-readable explanations; falls back to rule-based summaries if the API is unavailable |
| `src/alert_manager.py` | Formats and dispatches data-quality alerts |
| `api/main.py`, `api/schemas.py` | FastAPI service + request/response models |
| `dags/quality_monitor_dag.py` | Airflow DAG that schedules the daily monitoring run |
| `tests/` | Unit tests for the profiler and anomaly detector |

## Anomaly Checks

| Check | Rule | Severity |
| --- | --- | --- |
| Null rate | > 5% warning, > 20% critical | medium / critical |
| Z-score outlier | column min or max more than 3 standard deviations from the mean | medium |
| IQR outlier | values outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR] | low |
| Mean drift | column mean moved more than 3 baseline standard deviations since the previous run | high |
| Volume drift | row count > 30% below the 7-day average (critical) or > 2x above it (high, possible duplication) | critical / high |
| Schema change | column added (medium) or removed (critical) since the previous run | medium / critical |

Mean drift and schema checks compare against the previous profile, stored as JSON in `PROFILE_STORE_DIR` (default `profiles/`). On the first run there is no baseline, so those two checks are skipped.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 |
| Warehouse | Google BigQuery |
| LLM | OpenAI (`gpt-4o-mini` default, configurable) + rule-based fallback |
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

Python | Google BigQuery | Data Profiling | Statistical Anomaly Detection | OpenAI API / LLM | FastAPI | Apache Airflow | Docker | MLOps | Data Quality
