"""Pydantic request/response schemas for the FastAPI application."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class MonitorRequest(BaseModel):
    project_id: str = Field(..., description="GCP project ID")
    dataset: str    = Field(..., description="BigQuery dataset name")
    table: str      = Field(..., description="Table to monitor")
    lookback_days: int = Field(7, ge=1, le=90, description="Days of history for drift checks")
    null_rate_threshold: float = Field(0.05, ge=0.0, le=1.0, description="Null rate that triggers a warning")
    date_column: str = Field("created_at", description="Date/timestamp column for daily row counts")


class AnomalyDetail(BaseModel):
    anomaly_type: str
    column: str | None
    description: str
    severity: str


class MonitorResponse(BaseModel):
    table: str
    run_id: str
    anomalies_found: int
    critical_count: int
    anomalies: list[AnomalyDetail]
    llm_explanation: str
    alert_channels: dict[str, bool]
    profiled_at: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
