from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from core.schemas import CaseResult, EvalRun, EvalRunVersion
from core.store import (
    DEFAULT_DATABASE_URL,
    Base,
    DuplicateRunError,
    RunNotFoundError,
    Store,
    database_url_from_env,
    normalize_database_url,
)

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def make_run(
    *,
    run_id: str | None = None,
    ts: datetime = T0,
    harness: str = "h1",
    model: str = "m1",
    grader: str = "g1",
    dataset: str = "d1",
    n_cases: int = 3,
    pinned: bool = False,
) -> EvalRun:
    cases = [
        CaseResult(
            case_id=f"case-{i}",
            task_type="search" if i % 2 == 0 else "issues",
            score=(i % 3) / 2,
            passed=(i % 3) == 2,
            raw_trace_ref=f"s3://traces/{i}",
            tools_used=["search_repos", "get_issue"][: (i % 2) + 1],
            latency_ms=100.5 + i,
            output_preview=f"output {i}",
        )
        for i in range(n_cases)
    ]
    kwargs = {} if run_id is None else {"run_id": run_id}
    return EvalRun(
        version=EvalRunVersion(
            harness_version=harness,
            model_version=model,
            grader_version=grader,
            dataset_version=dataset,
            timestamp=ts,
        ),
        cases=cases,
        cases_attempted=n_cases,
        is_pinned_control=pinned,
        **kwargs,
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_runs.db"


@pytest.fixture
def store(db_path: Path):
    s = Store(f"sqlite:///{db_path}")
    yield s
    s.dispose()


# ---------------------------------------------------------------- round trip
def test_roundtrip_equality(store: Store) -> None:
    run = make_run()
    assert store.save_run(run) == run.run_id
    assert store.get_run(run.run_id) == run


def test_roundtrip_preserves_case_order_and_optional_fields(store: Store) -> None:
    run = make_run(n_cases=12)
    store.save_run(run)
    loaded = store.get_run(run.run_id)
    assert [c.case_id for c in loaded.cases] == [c.case_id for c in run.cases]
    assert loaded.cases[3].tools_used == run.cases[3].tools_used
    assert loaded.cases[3].latency_ms == run.cases[3].latency_ms
    assert loaded.cases_attempted == 12


def test_roundtrip_none_fields(store: Store) -> None:
    run = EvalRun(
        version=make_run().version,
        cases=[CaseResult(case_id="a", task_type="t", score=0.25, passed=False)],
    )
    store.save_run(run)
    loaded = store.get_run(run.run_id)
    assert loaded == run
    assert loaded.cases_attempted is None
    assert loaded.cases[0].raw_trace_ref is None


def test_empty_run_roundtrips_and_has_no_aggregate(store: Store) -> None:
    run = EvalRun(version=make_run().version, cases=[], cases_attempted=30)
    store.save_run(run)
    loaded = store.get_run(run.run_id)
    assert loaded == run
    assert loaded.aggregate_score is None  # never a fabricated 0.0


def test_timestamp_is_normalized_to_utc(store: Store) -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    run = make_run(ts=datetime(2026, 10, 1, 17, 30, tzinfo=ist))
    store.save_run(run)
    loaded = store.get_run(run.run_id)
    assert loaded.version.timestamp == T0  # same instant
    assert loaded.version.timestamp.utcoffset() == timedelta(0)  # tz re-attached after SQLite


# ------------------------------------------------------------- immutability
def test_duplicate_run_id_rejected_and_original_untouched(store: Store) -> None:
    original = make_run(run_id="dup", n_cases=3)
    store.save_run(original)
    with pytest.raises(DuplicateRunError):
        store.save_run(make_run(run_id="dup", n_cases=5, grader="g2"))
    assert store.get_run("dup") == original  # no overwrite, no partial extra cases


def test_get_missing_run_raises(store: Store) -> None:
    with pytest.raises(RunNotFoundError):
        store.get_run("nope")


# ------------------------------------------------------------------ listing
def test_list_runs_ordering_and_limit(store: Store) -> None:
    runs = [make_run(run_id=f"r{i}", ts=T0 + timedelta(days=i)) for i in range(5)]
    for r in reversed(runs):  # insert out of order
        store.save_run(r)
    assert [r.run_id for r in store.list_runs(order="asc")] == [f"r{i}" for i in range(5)]
    assert [r.run_id for r in store.list_runs()] == [f"r{i}" for i in reversed(range(5))]
    assert [r.run_id for r in store.list_runs(limit=2)] == ["r4", "r3"]
    assert store.list_runs(limit=0) == []
    with pytest.raises(ValueError):
        store.list_runs(limit=-1)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("harness_version", "h2", {"b"}),
        ("model_version", "m2", {"c"}),
        ("grader_version", "g2", {"d"}),
        ("dataset_version", "d2", {"e"}),
        ("harness_version", "h1", {"a", "c", "d", "e"}),
    ],
)
def test_list_runs_filters_by_each_version_field(
    store: Store, field: str, value: str, expected: set[str]
) -> None:
    store.save_run(make_run(run_id="a"))
    store.save_run(make_run(run_id="b", harness="h2"))
    store.save_run(make_run(run_id="c", model="m2"))
    store.save_run(make_run(run_id="d", grader="g2"))
    store.save_run(make_run(run_id="e", dataset="d2"))
    got = {r.run_id for r in store.list_runs(**{field: value})}
    assert got == expected


def test_list_runs_combined_filters_and_time_window(store: Store) -> None:
    for i in range(4):
        store.save_run(make_run(run_id=f"r{i}", ts=T0 + timedelta(days=i), model="m1"))
    store.save_run(make_run(run_id="other", ts=T0 + timedelta(days=1), model="m2"))
    got = store.list_runs(
        model_version="m1", since=T0 + timedelta(days=1), until=T0 + timedelta(days=2), order="asc"
    )
    assert [r.run_id for r in got] == ["r1", "r2"]


# ----------------------------------------------------------- pinned control
def test_pinned_control_runs(store: Store) -> None:
    store.save_run(make_run(run_id="normal"))
    store.save_run(make_run(run_id="ctl2", pinned=True, ts=T0 + timedelta(days=2)))
    store.save_run(make_run(run_id="ctl1", pinned=True, ts=T0 + timedelta(days=1)))
    pinned = store.get_pinned_control_runs()
    assert [r.run_id for r in pinned] == ["ctl1", "ctl2"]
    assert all(r.is_pinned_control for r in pinned)


def test_no_pinned_runs_returns_empty(store: Store) -> None:
    store.save_run(make_run())
    assert store.get_pinned_control_runs() == []


# ------------------------------------------------- statelessness / restarts
def test_data_survives_new_store_instance(db_path: Path) -> None:
    """A fresh Store on the same URL (a 'restart') sees everything: no in-memory state."""
    url = f"sqlite:///{db_path}"
    run = make_run(run_id="persist")
    s1 = Store(url)
    s1.save_run(run)
    s1.dispose()
    s2 = Store(url)
    try:
        assert s2.get_run("persist") == run
    finally:
        s2.dispose()


# ------------------------------------------------------- config / portability
def test_default_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database_url_from_env() == DEFAULT_DATABASE_URL == "sqlite:///./eval_runs.db"


def test_database_url_from_env(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    s = Store()  # no explicit URL: must pick up the env var
    try:
        s.save_run(make_run(run_id="env"))
        assert db_path.exists()
    finally:
        s.dispose()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("postgres://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        ("postgresql://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("sqlite:///./x.db", "sqlite:///./x.db"),
    ],
)
def test_normalize_database_url(raw: str, expected: str) -> None:
    assert normalize_database_url(raw) == expected


def test_schema_compiles_for_postgres() -> None:
    """No live Postgres here, so check the DDL at least compiles on that dialect."""
    ddl = "\n".join(
        str(CreateTable(t).compile(dialect=postgresql.dialect()))
        for t in Base.metadata.sorted_tables
    )
    assert "eval_runs" in ddl and "case_results" in ddl
    assert "JSON" in ddl and "TIMESTAMP WITH TIME ZONE" in ddl


def test_repr_never_exposes_database_url(db_path: Path) -> None:
    """repr (which ends up in logs/tracebacks) must not contain the URL or file path."""
    s = Store(f"sqlite:///{db_path}")
    try:
        assert str(db_path) not in repr(s)
        assert repr(s) == "Store(dialect='sqlite')"
    finally:
        s.dispose()