"""R3A.1 F — rotating-compression conversation-identity integration test.

Required functional invariant, exercised through the REAL production
rotation path (never by directly assigning ``_SESSION_ID``):

    ROOT-A / physical SID-A
    -> publish conversation id ROOT-A
    -> REAL rotating compression transition SID-A -> SID-B
    -> physical session id becomes SID-B
    -> logical HERMES_CONVERSATION_ID remains ROOT-A
    -> subprocess created after the transition sees:
         HERMES_SESSION_ID      = SID-B
         HERMES_CONVERSATION_ID = ROOT-A

The rotation runs through the production engine entry point
``AIAgent._compress_context`` with the summarizer stubbed — the LLM summary
is the engine's ONLY external dependency; every lock acquisition, DB
transition, child publication and caller-side ContextVar repair below
executes the real production code. (The harness is the one F5 of
``tests/agent/test_compression_worker_isolation_76354.py`` established for
the same engine; it runs hermetically, offline.)

ContextVar thread-isolation note — why NO worker re-sync hook exists:
the compression engine may run its summarizer on an owned pooled WORKER
thread (``run_compress_context_with_progress_timeout``), and ContextVars
are thread- AND task-local: a publication performed on a worker thread
cannot propagate to the caller's context (a detached asyncio task cannot
either). Because the compression-lineage ROOT is rotation-invariant by
construction, the caller's pre-compression prologue binding ROOT-A is
ALREADY the correct post-rotation value. A re-sync therefore cannot add
information in any consumer-visible topology: on same-thread paths it would
re-bind the identical value, on worker paths its write is dropped. The
R3A-era re-sync hook in ``agent/conversation_compression.py`` was REMOVED
as redundant (restoring the audited base bytes) rather than kept as a
misleading hook; this test pins the invariant that removal relies on.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

CONVERSATION_ENV = "HERMES_CONVERSATION_ID"
SESSION_ENV = "HERMES_SESSION_ID"


def _build_agent_with_db(db, session_id: str):
    """Real AIAgent over the REAL db (same construction F3-F5 of
    tests/agent/test_compression_worker_isolation_76354.py use)."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor._last_compression_made_progress = True
    compressor._last_summary_fallback_used = False
    agent.context_compressor = compressor
    # The compressor is a stub — the one-time compression-model feasibility
    # probe would resolve a REAL auxiliary provider before the engine runs;
    # skipping it keeps the suite hermetic (mirrors the 76354 harness).
    agent._compression_feasibility_checked = True
    return agent


def _rotate_db(db, parent_id: str, child_id: str) -> None:
    """Real DB-level rotating compression transition (compression lock +
    atomic ``publish_compression_child`` — the production-only way to mint
    a compression continuation)."""
    holder = f"r3a1:{parent_id}:{child_id}"
    assert db.try_acquire_compression_lock(parent_id, holder)
    db.publish_compression_child(
        parent_session_id=parent_id,
        child_session_id=child_id,
        source="telegram",
        messages=[{"role": "user", "content": "handoff"}],
        model="test/model",
        compression_lock_holder=holder,
    )
    db.release_compression_lock(parent_id, holder)


def test_rotating_compression_identity_invariant(tmp_path, monkeypatch):
    """ROOT-A / SID-A -> publish ROOT-A -> real rotation SID-A -> SID-B ->
    physical id SID-B, logical id ROOT-A, subprocess env carries both."""
    from gateway.session_context import (
        clear_session_vars,
        get_session_env,
        set_session_vars,
    )
    from hermes_state import SessionDB
    from tools.environments.local import _make_run_env

    from agent.conversation_identity import (
        publish_conversation_identity_for_agent,
        resolve_conversation_identity,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    root_a, sid_a = "ROOT-A", "SID-A"
    db.create_session(root_a, source="telegram")
    _rotate_db(db, root_a, sid_a)  # physical SID-A, logical root ROOT-A

    # Real agent on the real DB at physical SID-A; legacy ROTATION mode.
    agent = _build_agent_with_db(db, sid_a)
    agent.compression_in_place = False
    agent._cached_system_prompt = "sys"
    # Owned pooled wrapper: the summarizer MAY run on a worker thread — the
    # caller's ContextVars can only be repaired by the caller (F5 contract).
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (5.0, 10.0),
    )

    # Gateway-style caller binding + the per-turn prologue publication
    # (the exact runtime publication site of agent/turn_context).
    tokens = set_session_vars(session_id=sid_a, platform="telegram")
    try:
        published = publish_conversation_identity_for_agent(agent)
        assert published == root_a
        assert get_session_env(CONVERSATION_ENV) == root_a
        assert get_session_env(SESSION_ENV) == sid_a

        # ── REAL rotating compression transition SID-A -> SID-B ──
        messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(messages, "sys", approx_tokens=120_000)

        sid_b = agent.session_id
        assert sid_b != sid_a, "the engine must have minted a new physical id"
        # DB truth: the continuation row exists and still resolves to ROOT-A.
        assert db.get_session(sid_b) is not None
        assert resolve_conversation_identity(sid_b, db) == root_a
        # Physical id repaired on the CALLER (production repair helper).
        assert get_session_env(SESSION_ENV) == sid_b
        # ── REQUIRED functional invariant: logical identity stays ROOT-A ──
        assert get_session_env(CONVERSATION_ENV) == root_a, (
            "rotation must not change the canonical conversation identity"
        )
        # Subprocess env created AFTER the transition carries both values.
        env = _make_run_env({})
        assert env[SESSION_ENV] == sid_b
        assert env[CONVERSATION_ENV] == root_a
    finally:
        clear_session_vars(tokens)
