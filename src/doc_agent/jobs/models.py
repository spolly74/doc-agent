"""Job queue data models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a job in the queue."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobConfig(BaseModel):
    """Configuration for an evaluation job."""

    target_path: str
    output_path: str
    provider: str = "claude"
    model: str = "claude-sonnet-4-20250514"
    concurrency: int = 2
    mode: str = "full"  # "full" or "tiered"
    threshold: float = 4.0


class Job(BaseModel):
    """Represents an evaluation job in the queue."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(default=JobStatus.PENDING)
    config: JobConfig = Field(..., description="Job configuration")

    # Progress tracking
    progress_current: int = Field(default=0, description="Articles processed")
    progress_total: int = Field(default=0, description="Total articles to process")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    # Results
    error_message: str | None = Field(default=None)
    result_summary: dict[str, Any] | None = Field(default=None)

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.progress_total == 0:
            return 0.0
        return (self.progress_current / self.progress_total) * 100

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate job duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()
