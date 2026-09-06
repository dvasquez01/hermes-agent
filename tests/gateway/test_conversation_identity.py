"""HERMES_CONVERSATION_ID — canonical logical conversation lifecycle id.

Generic Hermes runtime primitive. The value is the
fork-aware compression-lineage ROOT of the current physical session id,
resolved through the EXISTING SessionDB.get_compression_lineage() semantics
(shared strict helper in agent/prompt_cache_scope — the walker is never
reimplemented here) and published per turn by the agent prologue into the
task-local ContextVar bridged to tool subprocesses via the official
_inject_session_context_env mechanism.

Lifecycle contracts exercised below (production revision audited):
  - ordinary turns, in-place compaction, and rotating compression splits all
    keep ONE identity (the root is rotation-invariant by construction);
  - /new-style boundaries (reset lineage) mint a new identity;
  - explicit forks (/branch, delegate, tool) are isolated by the
    compression-child predicate;
  - an unavailable/indeterminate lineage publishes NO identity (strict,
    never a guessed physical session id);
  - the bridge is ContextVar-authoritative once engaged, and the long-lived
    shell snapshot can never capture or replay the value.
"""

import asyncio
import os
import sys
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gateway.session_context as sc  # noqa: E402
from gateway.session_context import (  # noqa: E402
    _VAR_MAP,
    clear_session_vars,
    get_session_env,
    reset_session_vars,
    set_session_vars,
)
from tools.environments.base import (  # noqa: E402
    _SNAPSHOT_EXCLUDED_ENV_REGEX,
    _export_dump_excluding_session_vars,
)
from tools.environments.local import _make_run_env  # noqa: E402

CONVERSATION_ENV = "HERMES_CONVERSATION_ID"

SESSION_DB_KWARGS = {
    "source": "telegram",
    "session_key": "agent:main:telegram:dm:12345",
    "chat_id": "12345",
    "chat_type": "dm",
}

USER_HANDOFF = [{"role": "user", "content": "handoff", "_persist_disabled": True}]


@pytest.fixture
def session_db(tmp_path):
    from hermes_state import SessionDB

    return SessionDB(tmp_path / "state.db")


def _rotate(session_db, parent_id: str, child_id: str) -> None:
    """Compression rotation through the real production API (lock + atomic
    publish — the only supported way to mint a compression continuation)."""
    holder = f"r3a:{parent_id}:{child_id}"
    assert session_db.try_acquire_compression_lock(parent_id, holder)
    session_db.publish_compression_child(
        parent_session_id=parent_id,
        child_session_id=child_id,
        source=SESSION_DB_KWARGS["source"],
        messages=list(USER_HANDOFF),
        model="test-model",
        compression_lock_holder=holder,
    )
    session_db.release_compression_lock(parent_id, holder)


def _new_lineage(session_db, ended_id: str, fresh_id: str) -> None:
    """/new boundary DB semantics: promote the live tip to a session_reset
    end and create the fresh parentless row (gateway/session.py
    reset_session contract)."""
    assert session_db.promote_to_session_reset(ended_id, "session_reset")
    session_db.create_session(fresh_id, model_config={"_reset_from": ended_id},
                              **SESSION_DB_KWARGS)


class _Agent:
    def __init__(self, session_id, session_db):
        self.session_id = session_id
        self._session_db = session_db


def _publish(agent) -> None:
    from agent.conversation_identity import publish_conversation_identity_for_agent

    publish_conversation_identity_for_agent(agent)


# ── strict resolver (pure SessionDB scope) ─────────────────────────────────


def test_strict_root_resolves_lineage_root(session_db):
    from agent.conversation_identity import resolve_conversation_identity

    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    _rotate(session_db, "rootA", "segB")
    assert resolve_conversation_identity("segB", session_db) == "rootA"


def test_strict_root_never_guesses_on_missing_row(session_db):
    """A session whose row is absent (not yet persisted / transient failure)
    publishes NO identity — the physical id is never laundered into an
    authoritative-looking conversation id."""
    from agent.conversation_identity import resolve_conversation_identity

    assert resolve_conversation_identity("ghost123", session_db) == ""


def test_strict_root_never_raises(session_db):
    from agent.conversation_identity import resolve_conversation_identity

    assert resolve_conversation_identity("", session_db) == ""
    assert resolve_conversation_identity("x", None) == ""


def test_prompt_cache_scope_strict_variant_shares_the_walker(session_db):
    """The strict variant returns root-or-None and does NOT fall back to the
    physical id, while the legacy cache-scope fallback semantics stay intact
    (degrade to physical id on missing row)."""
    from agent.prompt_cache_scope import (
        resolve_prompt_cache_scope,
        resolve_prompt_cache_scope_strict,
    )

    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    _rotate(session_db, "rootA", "segB")
    agent = _Agent("segB", session_db)
    assert resolve_prompt_cache_scope_strict(agent) == "rootA"
    # legacy contract: same value when the lineage is resolvable
    assert resolve_prompt_cache_scope(agent) == "rootA"
    # legacy degrades to the physical id; strict must not
    agent_missing = _Agent("noRow", session_db)
    assert resolve_prompt_cache_scope(agent_missing) == "noRow"
    assert resolve_prompt_cache_scope_strict(agent_missing) is None


# ── lifecycle matrix (CID-01..11, 17) ─────────────────────────────────────


def test_cid01_ordinary_turns_preserve_identity(session_db):
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    _publish(_Agent("rootA", session_db))
    first = get_session_env(CONVERSATION_ENV)
    assert first == "rootA"
    # next turn, same conversation
    _publish(_Agent("rootA", session_db))
    assert get_session_env(CONVERSATION_ENV) == first


def test_cid02_in_place_compaction_preserves_identity(session_db):
    """archive_and_compact (default in_place mode) keeps the row live and
    identity-less-changed → conversation id constant."""
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    _publish(_Agent("rootA", session_db))
    before = get_session_env(CONVERSATION_ENV)
    session_db.archive_and_compact("rootA", list(USER_HANDOFF))
    _publish(_Agent("rootA", session_db))
    assert get_session_env(CONVERSATION_ENV) == before == "rootA"


def test_cid03_single_rotation_keeps_root(session_db):
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    _rotate(session_db, "rootA", "segB")
    _publish(_Agent("segB", session_db))
    assert get_session_env(CONVERSATION_ENV) == "rootA"


def test_cid04_multi_rotation_keeps_root(session_db):
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    _rotate(session_db, "rootA", "segB")
    _rotate(session_db, "segB", "segC")
    _publish(_Agent("segC", session_db))
    assert get_session_env(CONVERSATION_ENV) == "rootA"


def test_cid05_new_boundary_rotates_identity(session_db):
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    _publish(_Agent("rootA", session_db))
    _new_lineage(session_db, "rootA", "rootD")
    _publish(_Agent("rootD", session_db))
    assert get_session_env(CONVERSATION_ENV) == "rootD"


def test_cid06_reset_creates_new_root(session_db):
    """The /reset user boundary ends the live tip (session_reset /
    user_exit end_reasons) → the successor row is no longer a compression
    child → fresh identity."""
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    session_db.end_session("rootA", "user_exit")
    session_db.create_session("rootE", **SESSION_DB_KWARGS)  # parentless successor
    _publish(_Agent("rootE", session_db))
    assert get_session_env(CONVERSATION_ENV) == "rootE"
    assert get_session_env(CONVERSATION_ENV) != "rootA"


def test_cid07_branch_child_isolated(session_db):
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    session_db.create_session(
        "branchChild",
        parent_session_id="rootA",
        model_config={"_branched_from": "rootA"},
        **SESSION_DB_KWARGS,
    )
    _publish(_Agent("branchChild", session_db))
    assert get_session_env(CONVERSATION_ENV) == "branchChild"


def test_cid08_delegate_child_isolated(session_db):
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    session_db.create_session(
        "delegateChild",
        parent_session_id="rootA",
        model_config={"_delegate_from": "rootA"},
        **SESSION_DB_KWARGS,
    )
    _publish(_Agent("delegateChild", session_db))
    assert get_session_env(CONVERSATION_ENV) == "delegateChild"


def test_cid09_switch_resumes_other_lineage_root(session_db):
    """resume/switch activates a different conversation row; identity resolves
    to THAT row's own root (SID-9 rotation chain under rootB)."""
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    session_db.create_session("rootB", **SESSION_DB_KWARGS)
    _rotate(session_db, "rootB", "segB2")
    _publish(_Agent("rootA", session_db))
    assert get_session_env(CONVERSATION_ENV) == "rootA"
    _publish(_Agent("segB2", session_db))
    assert get_session_env(CONVERSATION_ENV) == "rootB"


def test_cid10_reopened_db_same_root(session_db, tmp_path):
    from hermes_state import SessionDB

    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    _rotate(session_db, "rootA", "segB")
    reopened = SessionDB(tmp_path / "state.db")
    assert _strict_root("segB", reopened) == "rootA"


def _strict_root(sid, db):
    from agent.conversation_identity import resolve_conversation_identity

    return resolve_conversation_identity(sid, db)


def test_cid11_unavailable_lineage_absent_not_guessed(session_db):
    """ContextVar engaged (gateway-style) but the agent cannot resolve an
    authoritative root (no DB / missing row) → identity published as EMPTY,
    never the physical session id."""
    reset_session_vars()
    set_session_vars(session_id="segMissing", session_key="k")
    # no DB at all
    _publish(_Agent("segMissing", None))
    assert get_session_env(CONVERSATION_ENV) == ""
    # DB present but row absent
    _publish(_Agent("segMissing", session_db))
    assert get_session_env(CONVERSATION_ENV) == ""


def test_cid17_normal_compression_child_continues_parent_root(session_db):
    session_db.create_session("rootA", **SESSION_DB_KWARGS)
    _rotate(session_db, "rootA", "segB")
    # the continuation has NO fork markers — it inherits the parent root
    row = session_db.get_session("segB")
    assert row["end_reason"] is None
    assert row["parent_session_id"] == "rootA"
    assert _strict_root("segB", session_db) == "rootA"


# ── subprocess bridge + leak guards (CID-12..15) ──────────────────────────


@pytest.fixture(autouse=True)
def _isolate_session_context():
    saved_env = {k: os.environ.get(k) for k in _VAR_MAP}
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    saved_engaged = sc._session_context_engaged
    for var in _VAR_MAP.values():
        var.set(sc._UNSET)
    sc._session_context_engaged = False
    try:
        yield
    finally:
        for name, var in _VAR_MAP.items():
            var.set(saved_ctx[name])
        sc._session_context_engaged = saved_engaged
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cid12_child_subprocess_env_exact_current_value():
    reset_session_vars()
    set_session_vars(session_key="kA", session_id="segA")
    sc._SESSION_CONVERSATION_ID.set("rootA")
    out = _make_run_env({})
    assert out[CONVERSATION_ENV] == "rootA"
    # rotation: mid-conversation child sees the SAME logical id although the
    # physical id changed
    sc._SESSION_ID.set("segB")
    out2 = _make_run_env({})
    assert out2["HERMES_SESSION_ID"] == "segB"
    assert out2[CONVERSATION_ENV] == "rootA"


def test_cid13_two_sessions_concurrent_no_contamination():
    """Two bound tasks resolve + publish independently; each sees its own
    conversation id (task-local ContextVar semantics)."""
    from hermes_state import SessionDB
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        db = SessionDB(Path(d) / "state.db")
        db.create_session("rootA", **SESSION_DB_KWARGS)
        db.create_session("rootZ", **SESSION_DB_KWARGS)
        _rotate(db, "rootZ", "segZ")

        async def worker(agent_sid, expect):
            async def run():
                reset_session_vars()
                set_session_vars(session_key=f"k-{agent_sid}", session_id=agent_sid)
                _publish(_Agent(agent_sid, db))
                await asyncio.sleep(0)  # interleave tasks
                assert get_session_env(CONVERSATION_ENV) == expect
            return await asyncio.create_task(run())

        asyncio.run(worker("rootA", "rootA"))
        asyncio.run(worker("segZ", "rootZ"))


def test_cid14_clear_and_reset_mask_conversation_id():
    reset_session_vars()
    set_session_vars(session_key="k", session_id="segA")
    sc._SESSION_CONVERSATION_ID.set("rootA")
    assert get_session_env(CONVERSATION_ENV) == "rootA"
    clear_session_vars([])
    assert get_session_env(CONVERSATION_ENV) == ""
    # reset returns to never-bound; with no host engaged the legacy os.environ
    # fallback applies (CLI one-shot), so a leaked global must be masked the
    # moment the machinery re-engages (CID-15 below).
    reset_session_vars()
    assert _VAR_MAP[CONVERSATION_ENV].get() is sc._UNSET


def test_cid15_foreign_os_environ_cannot_override_engaged_context():
    reset_session_vars()
    # session B's value stuck in the process-global mirror (last-writer-wins)
    os.environ[CONVERSATION_ENV] = "rootB-leak"
    os.environ["HERMES_SESSION_ID"] = "segB-leak"
    # session A binds its own trusted context WITHOUT a conversation id yet
    set_session_vars(session_key="kA", session_id="segA")
    out = _make_run_env({})
    # ContextVar explicitly "" at bind → foreign global stripped/overridden
    assert out.get(CONVERSATION_ENV, "") == ""
    assert out.get("HERMES_SESSION_ID") == "segA"
    # and when A resolves its own root, A's value wins authoritatively
    sc._SESSION_CONVERSATION_ID.set("rootA")
    out2 = _make_run_env({})
    assert out2[CONVERSATION_ENV] == "rootA"


def test_snapshot_exclusion_contract_covers_conversation_id():
    rx = __import__("re").compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    line = f'declare -x {CONVERSATION_ENV}="rootA"'
    assert rx.search(line), f"{CONVERSATION_ENV} must never enter a snapshot"
    snippet = _export_dump_excluding_session_vars('"$__hermes_snap_tmp"')
    assert CONVERSATION_ENV in snippet
    assert "grep -vE" not in snippet


def test_bridge_strips_when_unset_and_engaged():
    reset_session_vars()
    os.environ[CONVERSATION_ENV] = "rootB-leak"
    set_session_vars(session_key="kB", session_id="segB")
    # simulate an inherited-context task that reset BEFORE binding
    reset_session_vars()
    assert sc._session_context_engaged is True
    out = _make_run_env({CONVERSATION_ENV: "rootB-leak"})
    assert CONVERSATION_ENV not in out
