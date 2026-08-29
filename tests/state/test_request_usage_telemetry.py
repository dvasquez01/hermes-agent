"""Request-level usage telemetry contracts."""

import sqlite3
from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from agent.aux_accounting import reset_accounting_context, set_accounting_context
from agent.auxiliary_client import _validate_llm_response
from hermes_state import SCHEMA_VERSION, SessionDB


def _columns(db, table):
    return [row["name"] for row in db._conn.execute(f'PRAGMA table_info("{table}")')]


def test_request_usage_schema_is_current_and_private(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert SCHEMA_VERSION == 27
        assert _columns(db, "request_usage") == [
            "id", "session_id", "timestamp", "provider", "model", "task",
            "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "cache_read_tokens", "cache_write_tokens", "cache_miss_tokens",
            "input_cost_usd", "output_cost_usd", "total_cost_usd",
            "compression_generation", "event_type",
        ]
        names = {row[1] for row in db._conn.execute("PRAGMA index_list('request_usage')")}
        assert "idx_request_usage_session_timestamp" in names
        assert "idx_request_usage_provider_model_timestamp" in names
        assert not {"prompt", "content", "reasoning", "tool_output", "retrieved_memory"} & set(_columns(db, "request_usage"))
    finally:
        db.close()


def test_request_usage_is_one_row_per_queued_request(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        for generation, prompt, hit in ((0, 10000, 9000), (0, 10000, 0)):
            db.queue_token_counts(
                "s1",
                input_tokens=prompt - hit,
                output_tokens=100,
                cache_read_tokens=hit,
                model="deepseek-v4-flash",
                billing_provider="deepseek",
                api_call_count=1,
                estimated_cost_usd=0.001,
                request_usage={
                    "provider": "deepseek", "model": "deepseek-v4-flash",
                    "task": "normal", "event_type": "normal",
                    "prompt_tokens": prompt, "completion_tokens": 100,
                    "reasoning_tokens": 0, "cache_read_tokens": hit,
                    "cache_write_tokens": 0, "cache_miss_tokens": prompt - hit,
                    "input_cost_usd": 0.0001, "output_cost_usd": 0.0002,
                    "total_cost_usd": 0.001,
                    "compression_generation": generation,
                },
            )
        assert db.flush_token_counts()
        rows = db._conn.execute(
            "SELECT prompt_tokens, cache_read_tokens, cache_miss_tokens, "
            "compression_generation FROM request_usage WHERE session_id='s1' "
            "ORDER BY id"
        ).fetchall()
        assert [tuple(row) for row in rows] == [(10000, 9000, 1000, 0), (10000, 0, 10000, 0)]
        assert rows[0][1] / (rows[0][1] + rows[0][2]) == 0.9
        aggregate = db._conn.execute(
            "SELECT estimated_cost_usd FROM sessions WHERE id='s1'"
        ).fetchone()[0]
        request_total = db._conn.execute(
            "SELECT SUM(total_cost_usd) FROM request_usage WHERE session_id='s1'"
        ).fetchone()[0]
        assert request_total == aggregate == 0.002
    finally:
        db.close()


def test_compression_response_records_current_generation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        compressor = ContextCompressor.__new__(ContextCompressor)
        compressor._session_db = db
        compressor._session_id = "s1"
        compressor.base_url = ""
        compressor.compression_count = 1
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10000,
                completion_tokens=200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=9000),
            )
        )
        compressor._record_compression_request_usage(
            response, provider="deepseek", model="deepseek-v4-flash"
        )
        row = db._conn.execute(
            "SELECT task, event_type, prompt_tokens, cache_read_tokens, "
            "cache_miss_tokens, compression_generation FROM request_usage"
        ).fetchone()
        assert tuple(row) == ("compression", "compression", 10000, 9000, 1000, 1)
    finally:
        db.close()


def test_compression_validation_and_explicit_writer_produce_one_row(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        response = SimpleNamespace(
            model="deepseek-v4-flash",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))],
        )
        compressor = ContextCompressor.__new__(ContextCompressor)
        compressor._session_db = db
        compressor._session_id = "s1"
        compressor.base_url = ""
        compressor.compression_count = 1
        token = set_accounting_context(db, "s1")
        try:
            _validate_llm_response(response, task="compression", provider="deepseek")
        finally:
            reset_accounting_context(token)
        compressor._record_compression_request_usage(
            response, provider="deepseek", model="deepseek-v4-flash"
        )
        assert db._conn.execute(
            "SELECT count(*) FROM request_usage WHERE session_id='s1'"
        ).fetchone()[0] == 1
        assert db._conn.execute(
            "SELECT sum(api_call_count) FROM session_model_usage WHERE session_id='s1'"
        ).fetchone()[0] == 1
        row = db._conn.execute(
            "SELECT task, event_type, compression_generation FROM request_usage"
        ).fetchone()
        assert tuple(row) == ("compression", "compression", 1)
    finally:
        db.close()


def test_auxiliary_usage_and_request_telemetry_commit_atomically(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        db.record_auxiliary_usage(
            "s1", "title", model="m", billing_provider="p",
            input_tokens=10, output_tokens=2, estimated_cost_usd=0.5,
            request_usage={
                "provider": "p", "model": "m", "task": "title",
                "prompt_tokens": 10, "completion_tokens": 2,
                "cache_miss_tokens": 10, "total_cost_usd": 0.5,
            },
        )
        assert db._conn.execute(
            "SELECT count(*) FROM request_usage WHERE session_id='s1'"
        ).fetchone()[0] == 1
        assert db._conn.execute(
            "SELECT sum(api_call_count) FROM session_model_usage WHERE session_id='s1'"
        ).fetchone()[0] == 1
        assert db._conn.execute(
            "SELECT sum(total_cost_usd) FROM request_usage WHERE session_id='s1'"
        ).fetchone()[0] == 0.5
    finally:
        db.close()


def test_compression_generations_are_explicit_and_multiple_boundaries_survive(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        sequence = [("normal", 0), ("normal", 0), ("compression", 0),
                    ("post_compression", 1), ("post_compression", 1),
                    ("compression", 1), ("post_compression", 2)]
        for task, generation in sequence:
            db.record_request_usage(
                "s1", provider="deepseek", model="deepseek-v4-flash", task=task,
                event_type=task, prompt_tokens=1, completion_tokens=1,
                reasoning_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
                cache_miss_tokens=1, input_cost_usd=0, output_cost_usd=0,
                total_cost_usd=0, compression_generation=generation,
            )
        rows = db._conn.execute(
            "SELECT task, compression_generation FROM request_usage ORDER BY id"
        ).fetchall()
        assert [tuple(row) for row in rows] == sequence
    finally:
        db.close()


def test_two_compressions_have_one_row_each_and_advance_generation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        compressor = ContextCompressor.__new__(ContextCompressor)
        compressor._session_db = db
        compressor._session_id = "s1"
        compressor.base_url = ""
        for generation in (0, 1):
            response = SimpleNamespace(
                model="deepseek-v4-flash",
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5),
                choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))],
            )
            compressor.compression_count = generation
            token = set_accounting_context(db, "s1")
            try:
                _validate_llm_response(response, task="compression", provider="deepseek")
            finally:
                reset_accounting_context(token)
            compressor._record_compression_request_usage(
                response, provider="deepseek", model="deepseek-v4-flash"
            )
        rows = db._conn.execute(
            "SELECT task, event_type, compression_generation FROM request_usage "
            "ORDER BY id"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("compression", "compression", 0),
            ("compression", "compression", 1),
        ]
    finally:
        db.close()


def test_auxiliary_usage_rolls_back_both_writes_on_telemetry_failure(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        try:
            db.record_auxiliary_usage(
                "s1", "title", model="m", billing_provider="p",
                input_tokens=10, output_tokens=2, estimated_cost_usd=0.5,
                request_usage={"prompt_tokens": 10, "compression_generation": "invalid"},
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid telemetry should fail")
        assert db._conn.execute(
            "SELECT count(*) FROM request_usage WHERE session_id='s1'"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT count(*) FROM session_model_usage WHERE session_id='s1'"
        ).fetchone()[0] == 0
    finally:
        db.close()