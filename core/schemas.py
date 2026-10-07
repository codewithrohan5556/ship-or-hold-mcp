"""Versioned eval-result schema.

This is the one contract the whole project depends on: the fault injector
writes it, the store persists it, the integrity detectors and the verdict
engine read it, and the MCP tools accept/return it.

Design rules:
- Every run carries the four versions that can silently change a score
  (harness, model, grader, dataset) plus a timestamp. Without them the
  "did the measurement break?" question is unanswerable.
- Scores live in [0, 1]. ``passed`` is stored separately because graders can
  change their pass threshold without changing the underlying score.
- Raw traces are never embedded: only a reference plus a short, truncated
  preview. Trace payloads can be arbitrarily large and must stay out of
  logs and tool responses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Cap on the stored output preview. Keeps rows small and keeps traces out of logs.
MAX_PREVIEW_CHARS = 500


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_run_id() -> str:
    return uuid4().hex


class EvalRunVersion(BaseModel):
    """Everything that can change a score without the agent changing."""

    model_config = ConfigDict(frozen=True)

    harness_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    grader_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=_utcnow)

    @field_validator("timestamp")
    @classmethod
    def _require_tz_aware(cls, v: datetime) -> datetime:
        # Naive datetimes make "did the score change line up with a deploy?"
        # comparisons ambiguous. Normalise everything to UTC.
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (use UTC)")
        return v.astimezone(UTC)


class CaseResult(BaseModel):
    """One scored case within a run."""

    case_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    # Pointer to the full raw trace (URI / key / path). The trace itself is never stored here.
    raw_trace_ref: str | None = None

    # --- Optional fields the Phase 5-6 detectors need ---------------------
    # Tools the agent invoked on this case (tool-error clustering).
    tools_used: list[str] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0.0)
    # Truncated raw output, so "score ~0 but output looks plausible" is checkable
    # without loading full traces.
    output_preview: str | None = None

    @field_validator("output_preview")
    @classmethod
    def _truncate_preview(cls, v: str | None) -> str | None:
        return v if v is None else v[:MAX_PREVIEW_CHARS]


class EvalRun(BaseModel):
    """A full scored eval run: version metadata plus per-case results."""

    run_id: str = Field(default_factory=_new_run_id, min_length=1)
    version: EvalRunVersion
    cases: list[CaseResult]
    # How many cases were actually attempted before scoring. If it differs from
    # len(cases), results were lost between attempt and score (denominator mismatch).
    # None means "not recorded", which is itself weaker evidence than a match.
    cases_attempted: int | None = Field(default=None, ge=0)
    # Tag for the frozen control set that is re-scored on every new grader version.
    is_pinned_control: bool = False

    @model_validator(mode="after")
    def _case_ids_unique(self) -> EvalRun:
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate case_id values in run: {dupes}")
        return self

    # --- Convenience, derived (never stored) ------------------------------
    @property
    def n_scored(self) -> int:
        return len(self.cases)

    @property
    def aggregate_score(self) -> float | None:
        """Mean case score, or None for an empty run (never a fabricated 0.0)."""
        if not self.cases:
            return None
        return sum(c.score for c in self.cases) / len(self.cases)

    @property
    def pass_rate(self) -> float | None:
        if not self.cases:
            return None
        return sum(c.passed for c in self.cases) / len(self.cases)