"""Canonical LOGICAL conversation identity for the current turn.

Generic Hermes runtime primitive. ``HERMES_CONVERSATION_ID`` is the ROOT of
the current physical session's *compression lineage*, resolved with the
STRICT fork-aware semantics of
``agent.prompt_cache_scope.resolve_prompt_cache_scope_strict`` (which shares
``SessionDB.get_compression_lineage`` — the walker is never reimplemented
here). This is deliberately NOT the Portal attribution root
(``SessionDB.get_conversation_root`` / ``run_agent._conversation_root_id``),
which follows ``parent_session_id`` blindly and collapses /branch and
delegate trees; the distinction is documented in
``agent/prompt_cache_scope`` and must not be "deduplicated".

Lifecycle (what the value guarantees to consumers):

- ordinary turns, in-place compaction, and rotating compression splits keep
  the SAME identity (the lineage root is rotation-invariant by construction);
- genuine conversation boundaries (/new, /reset, explicit /branch,
  delegate/tool fork, switch/resume into another lineage, exhausted- or
  recovery-reset) yield a DIFFERENT identity because they start a new
  compression lineage;
- the value is an opaque session-row id — no user content, no chat ids,
  no secrets;
- STRICT availability: when no authoritative root can be resolved (no
  SessionDB handle, the row has not been persisted yet, or the lineage walk
  fails) the identity is published as EMPTY. The physical session id is
  NEVER substituted as an authoritative-looking conversation id; a
  transient DB failure must not launder a rotating id into a canonical one.
  Consumers treat empty as "no canonical conversation identity" and must
  fail closed (operate sessionless).

Publication is exclusively runtime-internal (the per-turn prologue in
``agent/turn_context``). The compression-rotation path needs no re-sync:
the ROOT is rotation-invariant, so the prologue binding is already the
post-rotation value, and ContextVar thread/task locality would drop any
worker-thread write anyway (see the R3A.1 integration test in
``tests/agent/test_conversation_identity_rotation_integration.py``). No
tool, CLI argument, model output, or user text may set it. Task-local
ContextVar mechanics (and the subprocess-env bridge) live in
``gateway/session_context``: once the session machinery is engaged, a child
process only ever sees the value bound for THIS task.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def resolve_conversation_identity(session_id: str, session_db: Any) -> str:
    """Strict fork-aware compression-lineage root for *session_id*.

    Returns "" when the identity cannot be established authoritatively
    (no id, no DB handle, missing row, or a failed lineage walk). Never
    falls back to the physical session id.
    """
    if not session_id or session_db is None:
        return ""
    # Local import: keep this module import-cheap and avoid an
    # agent <-> gateway import cycle at module load.
    from agent.prompt_cache_scope import _lineage_root

    try:
        return _lineage_root(str(session_id), session_db) or ""
    except Exception:
        logger.debug("conversation identity lineage walk failed", exc_info=True)
        return ""


def resolve_conversation_identity_for_agent(agent: Any) -> str:
    """Strict variant over an agent-like object (``session_id`` +
    ``_session_db``); delegates to the single strict entry point in
    ``agent.prompt_cache_scope`` so cache-scope and conversation identity
    can never drift apart."""
    try:
        from agent.prompt_cache_scope import resolve_prompt_cache_scope_strict

        return resolve_prompt_cache_scope_strict(agent) or ""
    except Exception:
        logger.debug("agent conversation identity resolution failed", exc_info=True)
        return ""


def conversation_identity() -> str:
    """The current task's canonical conversation identity ("" = none)."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return ""
    return get_session_env("HERMES_CONVERSATION_ID", "") or ""


def publish_conversation_identity(conversation_id: str) -> None:
    """Bind the identity for the CURRENT context (task-local).

    Runtime-internal only. Intentionally never mirrors to ``os.environ``:
    the process-global mirror is a CLI/one-shot compatibility path, and a
    long-lived concurrent host leaking last-writer-wins conversation ids
    there is exactly the cross-session bug class the ContextVar bridge
    guards against. The subprocess-env bridge strips the var for tasks
    that never bound it (ContextVar ``_UNSET`` + engaged machinery).
    """
    from gateway.session_context import _SESSION_CONVERSATION_ID

    _SESSION_CONVERSATION_ID.set(str(conversation_id or ""))


def publish_conversation_identity_for_agent(agent: Any) -> str:
    """Resolve (strictly) and publish *agent*'s conversation identity.

    Called from the per-turn prologue once the session row is guaranteed
    persisted. Never raises: a resolution problem can only mean an EMPTY
    (sessionless) identity, which is the fail-closed outcome. Returns the
    published value for tests and callers that want it.
    """
    sid = str(getattr(agent, "session_id", None) or "")
    db = getattr(agent, "_session_db", None)
    identity = resolve_conversation_identity(sid, db)
    try:
        publish_conversation_identity(identity)
    except Exception:
        logger.debug("conversation identity publish failed", exc_info=True)
    return identity
