"""Versioned result store (SQLAlchemy; SQLite for dev, Postgres for FastMCP Cloud).

Design decisions worth knowing:

- **Append-only.** A saved run is immutable. Re-saving an existing ``run_id``
  raises ``DuplicateRunError`` rather than overwriting, because silently
  rewriting history would defeat the whole point of a *versioned* store.
- **Stateless.** The ``Store`` holds a connection pool, never data. Every
  call opens its own session/transaction, so nothing that matters lives in
  process memory (FastMCP Cloud instances are ephemeral).
- **Same code on both databases.** Only portable SQLAlchemy types are used
  (``JSON``, ``DateTime(timezone=True)``, ``Float``). SQLite drops tzinfo on
  read, so timestamps are always stored as UTC and re-attached on load.
- **No migrations.** Tables are created with ``create_all`` (idempotent).
  That is a deliberate scope cut, documented in the README.
- **Never leaks credentials.** ``DATABASE_URL`` may contain a password, so
  the URL is never logged or included in ``repr``.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
    select,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    selectinload,
    sessionmaker,
)

from core.schemas import CaseResult, EvalRun, EvalRunVersion

logger = logging.getLogger("eval_diagnostic.store")

DEFAULT_DATABASE_URL = "sqlite:///./eval_runs.db"


class DuplicateRunError(ValueError):
    """Raised when saving a run whose run_id already exists (runs are immutable)."""


class RunNotFoundError(KeyError):
    """Raised when a run_id is not in the store."""


# --------------------------------------------------------------------------
# URL handling
# --------------------------------------------------------------------------
def database_url_from_env() -> str:
    """DATABASE_URL env var, defaulting to local SQLite."""
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def normalize_database_url(url: str) -> str:
    """Make bare Postgres URLs use the installed psycopg (v3) driver.

    Supabase/Heroku-style URLs are ``postgres://`` or ``postgresql://``, which
    SQLAlchemy would route to psycopg2 (not installed). Explicit driver
    URLs (``postgresql+psycopg://``) pass through unchanged.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


# --------------------------------------------------------------------------
# ORM models (mirror core.schemas)
# --------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class EvalRunRow(Base):
    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    harness_version: Mapped[str] = mapped_column(String(128), index=True)
    model_version: Mapped[str] = mapped_column(String(128), index=True)
    grader_version: Mapped[str] = mapped_column(String(128), index=True)
    dataset_version: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    cases_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_pinned_control: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    cases: Mapped[list[CaseResultRow]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="CaseResultRow.position",
    )


class CaseResultRow(Base):
    __tablename__ = "case_results"
    __table_args__ = (UniqueConstraint("run_id", "case_id", name="uq_run_case"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.run_id"), index=True)
    # Preserves the original case order so a round trip is exactly equal.
    position: Mapped[int] = mapped_column(Integer)
    case_id: Mapped[str] = mapped_column(String(256))
    task_type: Mapped[str] = mapped_column(String(128))
    score: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)
    raw_trace_ref: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    tools_used: Mapped[list[str]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    output_preview: Mapped[str | None] = mapped_column(String(600), nullable=True)

    run: Mapped[EvalRunRow] = relationship(back_populates="cases")


# --------------------------------------------------------------------------
# Row <-> Pydantic conversion
# --------------------------------------------------------------------------
def _as_utc(dt: datetime) -> datetime:
    # SQLite returns naive datetimes; everything we store is UTC.
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _to_row(run: EvalRun) -> EvalRunRow:
    v = run.version
    return EvalRunRow(
        run_id=run.run_id,
        harness_version=v.harness_version,
        model_version=v.model_version,
        grader_version=v.grader_version,
        dataset_version=v.dataset_version,
        timestamp=v.timestamp,
        cases_attempted=run.cases_attempted,
        is_pinned_control=run.is_pinned_control,
        cases=[
            CaseResultRow(
                position=i,
                case_id=c.case_id,
                task_type=c.task_type,
                score=c.score,
                passed=c.passed,
                raw_trace_ref=c.raw_trace_ref,
                tools_used=list(c.tools_used),
                latency_ms=c.latency_ms,
                output_preview=c.output_preview,
            )
            for i, c in enumerate(run.cases)
        ],
    )


def _to_model(row: EvalRunRow) -> EvalRun:
    return EvalRun(
        run_id=row.run_id,
        version=EvalRunVersion(
            harness_version=row.harness_version,
            model_version=row.model_version,
            grader_version=row.grader_version,
            dataset_version=row.dataset_version,
            timestamp=_as_utc(row.timestamp),
        ),
        cases=[
            CaseResult(
                case_id=c.case_id,
                task_type=c.task_type,
                score=c.score,
                passed=c.passed,
                raw_trace_ref=c.raw_trace_ref,
                tools_used=list(c.tools_used or []),
                latency_ms=c.latency_ms,
                output_preview=c.output_preview,
            )
            for c in row.cases
        ],
        cases_attempted=row.cases_attempted,
        is_pinned_control=row.is_pinned_control,
    )


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
class Store:
    """Thin persistence layer over the versioned result tables."""

    def __init__(self, database_url: str | None = None) -> None:
        url = normalize_database_url(database_url or database_url_from_env())
        self._engine: Engine = create_engine(url, pool_pre_ping=True)
        if self._engine.dialect.name == "sqlite":
            _enable_sqlite_foreign_keys(self._engine)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)  # idempotent

    def __repr__(self) -> str:
        # Never include the URL: it may contain credentials.
        return f"Store(dialect={self._engine.dialect.name!r})"

    @property
    def dialect(self) -> str:
        return self._engine.dialect.name

    @property
    def safe_url(self) -> str:
        """URL with the password masked (safe for diagnostics)."""
        return make_url(str(self._engine.url)).render_as_string(hide_password=True)

    def dispose(self) -> None:
        self._engine.dispose()

    # -- writes ------------------------------------------------------------
    def save_run(self, run: EvalRun) -> str:
        """Persist a run atomically and return its run_id.

        Raises DuplicateRunError if the run_id already exists. Either the run
        and all its cases are stored, or nothing is.
        """
        with self._sessions() as session:
            if session.get(EvalRunRow, run.run_id) is not None:
                raise DuplicateRunError(f"run_id already exists: {run.run_id}")
            session.add(_to_row(run))
            try:
                session.commit()
            except IntegrityError as exc:  # lost a race with a concurrent writer
                session.rollback()
                raise DuplicateRunError(f"run_id already exists: {run.run_id}") from exc
        logger.info(
            "saved run run_id=%s n_cases=%d pinned=%s",
            run.run_id,
            len(run.cases),
            run.is_pinned_control,
        )
        return run.run_id

    # -- reads -------------------------------------------------------------
    def get_run(self, run_id: str) -> EvalRun:
        with self._sessions() as session:
            row = session.execute(
                select(EvalRunRow)
                .where(EvalRunRow.run_id == run_id)
                .options(selectinload(EvalRunRow.cases))
            ).scalar_one_or_none()
            if row is None:
                raise RunNotFoundError(run_id)
            return _to_model(row)

    def list_runs(
        self,
        *,
        harness_version: str | None = None,
        model_version: str | None = None,
        grader_version: str | None = None,
        dataset_version: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        order: Literal["asc", "desc"] = "desc",
        limit: int | None = 100,
    ) -> list[EvalRun]:
        """Runs matching every given filter, ordered by version timestamp.

        ``desc`` (default) is newest first, for "recent runs"; use ``asc``
        for chronological history (e.g. step-vs-slope shape fitting).
        """
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0 or None")
        stmt = select(EvalRunRow).options(selectinload(EvalRunRow.cases))
        filters = {
            EvalRunRow.harness_version: harness_version,
            EvalRunRow.model_version: model_version,
            EvalRunRow.grader_version: grader_version,
            EvalRunRow.dataset_version: dataset_version,
        }
        for column, value in filters.items():
            if value is not None:
                stmt = stmt.where(column == value)
        if since is not None:
            stmt = stmt.where(EvalRunRow.timestamp >= _as_utc(since))
        if until is not None:
            stmt = stmt.where(EvalRunRow.timestamp <= _as_utc(until))
        ts = EvalRunRow.timestamp
        # run_id as a tiebreaker keeps ordering deterministic for equal timestamps.
        stmt = stmt.order_by(ts.asc() if order == "asc" else ts.desc(), EvalRunRow.run_id)
        if limit is not None:
            stmt = stmt.limit(limit)
        with self._sessions() as session:
            return [_to_model(r) for r in session.execute(stmt).scalars().all()]

    def get_pinned_control_runs(self) -> list[EvalRun]:
        """The frozen control set, oldest first.

        These are re-scored on every new grader version: if their score moves
        while the subject is frozen, the *measurement* changed.
        """
        stmt = (
            select(EvalRunRow)
            .where(EvalRunRow.is_pinned_control.is_(True))
            .options(selectinload(EvalRunRow.cases))
            .order_by(EvalRunRow.timestamp.asc(), EvalRunRow.run_id)
        )
        with self._sessions() as session:
            return [_to_model(r) for r in session.execute(stmt).scalars().all()]


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """SQLite ignores foreign keys unless asked, on every new connection."""

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()