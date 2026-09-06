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
    """Two session tasks OVERLAP on ONE event loop; each sees its own
    conversation id in BOTH the ContextVar accessor (``get_session_env``)
    and the child env from the normal local subprocess-env builder
    (``tools.environments.local._make_run_env``).

    Deterministic adversarial overlap — asyncio.Event barriers, no sleeps
    as the synchronization mechanism:

    * Task A binds/publishes rootA, signals ``A_BOUND``, then BLOCKS on
      ``B_BOUND``. Task B (started by the same ``asyncio.gather``) waits
      for ``A_BOUND`` first, so its bind strictly OVERWRITES any shared
      global state AFTER A bound; B publishes rootZ (physical segZ), then
      signals ``B_BOUND`` and cooperatively yields.
    * A resumes while B is still bound-but-suspended inside its critical
      section and asserts A's OWN values — under a last-writer-wins /
      os.environ implementation A would observe B's rootZ/segZ here and
      fail. B then asserts its own values after A has run.

    Both tasks are alive simultaneously throughout (gather + event
    rendezvous), which the old sequential ``asyncio.run`` x2 version could
    never prove.
    """
    from hermes_state import SessionDB
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        db = SessionDB(Path(d) / "state.db")
        db.create_session("rootA", **SESSION_DB_KWARGS)
        db.create_session("rootZ", **SESSION_DB_KWARGS)
        _rotate(db, "rootZ", "segZ")  # physical segZ -> logical rootZ

        async def task_a(ev_a_bound, ev_b_bound):
            reset_session_vars()
            set_session_vars(session_key="k-rootA", session_id="rootA")
            _publish(_Agent("rootA", db))  # canonical root of lineage rootA
            ev_a_bound.set()
            # B binds strictly after A: any shared/global writer would have
            # overwritten A's values by the time A resumes.
            await ev_b_bound.wait()
            # ── A asserts while B is still bound inside its section ──
            assert get_session_env(CONVERSATION_ENV) == "rootA"
            assert get_session_env("HERMES_SESSION_ID") == "rootA"
            env_a = _make_run_env({})
            assert env_a[CONVERSATION_ENV] == "rootA"
            assert env_a["HERMES_SESSION_ID"] == "rootA"

        async def task_b(ev_a_bound, ev_b_bound):
            await ev_a_bound.wait()  # A has already bound + published rootA
            reset_session_vars()
            set_session_vars(session_key="k-segZ", session_id="segZ")
            _publish(_Agent("segZ", db))  # lineage segZ -> canonical rootZ
            ev_b_bound.set()
            await asyncio.sleep(0)  # cooperative yield: let A run mid-overlap
            assert get_session_env(CONVERSATION_ENV) == "rootZ"
            assert get_session_env("HERMES_SESSION_ID") == "segZ"
            env_b = _make_run_env({})
            assert env_b[CONVERSATION_ENV] == "rootZ"
            assert env_b["HERMES_SESSION_ID"] == "segZ"

        async def main():
            ev_a_bound, ev_b_bound = asyncio.Event(), asyncio.Event()
            await asyncio.gather(
                task_a(ev_a_bound, ev_b_bound),
                task_b(ev_a_bound, ev_b_bound),
            )

        asyncio.run(main())


def test_turn_prologue_publication_integration_fresh_session(tmp_path):
    """R3A.1 E — the PRODUCTION publication site (the hook added to
    ``agent/turn_context.build_turn_context``), not the publish helper alone.

    Required invariant, exercised through the real relevant turn setup path:

        fresh session -> REAL SessionDB row establishment (the agent's own
        ``_ensure_db_session`` inside the prologue) -> production turn-context
        publication -> HERMES_CONVERSATION_ID == canonical root -> child
        subprocess env (production local builder) carries the same root.

    A real ``AIAgent`` on a TEMPORARY SessionDB goes through the actual
    ``build_turn_context`` prologue; only the loop-injected callables that
    are irrelevant to row establishment + publication are stubbed. Temporary
    HERMES_HOME / temporary SessionDB — no live runtime, no network.
    """
    import types
    from unittest.mock import patch

    from agent.turn_context import build_turn_context
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    fresh_sid = "cidE-fresh-turn-session"
    assert db.get_session(fresh_sid) is None  # genuinely fresh: no row yet

    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "HERMES_HOME": str(tmp_path / "home")},
    ):
        (tmp_path / "home").mkdir(exist_ok=True)
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=fresh_sid,
            skip_context_files=True,
            skip_memory=True,
        )
    # Keep the prologue narrow: compression machinery stays off so the
    # publication site (which runs before idle/preflight compression) is
    # what this test exercises.
    agent.compression_enabled = False
    agent._cached_system_prompt = "SYSTEM"

    build_turn_context(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )

    # 1) Real SessionDB row establishment happened inside the prologue.
    row = db.get_session(fresh_sid)
    assert row is not None, "the turn prologue must establish the real session row"
    # 2) Canonical root of a fresh (parentless) lineage is its own row id.
    from agent.conversation_identity import resolve_conversation_identity

    root = resolve_conversation_identity(fresh_sid, db)
    assert root == fresh_sid
    # 3) The prologue's publication reached the task-local ContextVar...
    assert get_session_env(CONVERSATION_ENV) == root
    assert get_session_env("HERMES_SESSION_ID") == fresh_sid
    # 4) ...and the child subprocess env built by the production local
    #    subprocess-env builder carries the same canonical root.
    env = _make_run_env({})
    assert env[CONVERSATION_ENV] == root
    assert env["HERMES_SESSION_ID"] == fresh_sid


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
