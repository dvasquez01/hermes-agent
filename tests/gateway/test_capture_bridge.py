"""Conexión del dispatch autenticado al adapter de captura (memory-core).

Cubre la ruta insertada en GatewayRunner._handle_message — POST-AUTH,
POST-PAUSA y POST-CONTROLES de trabajo en curso, PRE-AGENTE:

- ruta explícita "captura …" y continuación de conversación activa;
- identidad canónica R3A por la cadena REAL del runtime
  (SessionStore.peek_session_id → resolve_conversation_identity sobre el
  SessionDB), estable tras compresión y separada tras reset/fork;
- sin mapping/fila/linaje: rechazo sanitizado (fail-closed), sin crear
  sesión y sin usar la clave de routing como identidad;
- pausa activa: cero invocaciones del puente para trabajo nuevo;
- trabajo en curso (agente en marcha, update/clarify/slash-confirm,
  aprobaciones): nunca se consume como captura;
- mensaje de captura sin message_id estable: rechazo sanitizado;
- fallos de lanzamiento/timeout/JSON: respuesta sanitizada con códigos
  controlados (sin stderr/stdout, texto del usuario, DSN ni payloads);
- autorización ANTES del routing (no autorizado → cero invocaciones);
- comandos (p. ej. /new) no se enrutan.

El E2E real (HERMES_CAPTURE_E2E=1) invoca el adapter REAL de memory-core
como subproceso contra la base TEST desechable, a través del MISMO dispatch,
con SessionStore/SessionDB reales (SQLite en tmp) para la identidad.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.capture_bridge import (
    CaptureBridgeFailure,
    CaptureIdentityUnavailable,
    CaptureMessageIdentityUnavailable,
    capture_bridge_config,
    is_capture_start,
    reset_active_captures,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB

CAPTURE_ENV = {
    "HERMES_CAPTURE_MEMORY_CORE_PYTHON": "python",
    "HERMES_CAPTURE_MEMORY_CORE_CWD": "cwd",
    "HERMES_CAPTURE_PROFILE": "capture-test",
}


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _enable_bridge(monkeypatch) -> None:
    for key, value in CAPTURE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MEMORY_CORE_DB_URL", raising=False)


def _disable_bridge(monkeypatch) -> None:
    for key in CAPTURE_ENV:
        monkeypatch.delenv(key, raising=False)


def _make_event(text: str = "captura decidí algo", *,
                platform: Platform = Platform.TELEGRAM,
                user_id: str = "12345",
                message_id: str = "m1",
                chat_id: str = "12345") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id=message_id,
        source=SessionSource(
            platform=platform,
            user_id=user_id,
            chat_id=chat_id,
            user_name="tester",
            chat_type="dm",
        ),
    )


@pytest.fixture()
def real_env(tmp_path, monkeypatch):
    """SessionStore + SessionDB REALES (SQLite en tmp; sin mocks).

    La frontera de identidad SOLO queda acreditada con la cadena real:
    mapping persistido en el store + fila/linaje reales en el SessionDB.
    """
    import hermes_state

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    db = SessionDB(db_path=tmp_path / "state.db")
    return store, db


def _seed_compression_chain(db: SessionDB, root: str, tip: str) -> None:
    """Cadena de compresión REAL (mismo camino que la producción:
    publish_compression_child con lock)."""
    db.create_session(root, source="telegram")
    holder = f"cc:{root}:{tip}"
    assert db.try_acquire_compression_lock(root, holder)
    db.publish_compression_child(
        parent_session_id=root,
        child_session_id=tip,
        source="telegram",
        messages=[{"role": "user", "content": "handoff"}],
        model="test/model",
        compression_lock_holder=holder,
    )
    db.release_compression_lock(root, holder)


def _bind_chain(store: SessionStore, db: SessionDB, source) -> tuple:
    """Asocia el source a una cadena REAL root→tip; devuelve (root, tip, key).

    El mapping del store apunta al TIP físico (estado normal tras una
    compresión en vuelo); la identidad canónica debe ser la RAÍZ.
    """
    root = f"root-{uuid.uuid4().hex[:10]}"
    tip = f"tip-{uuid.uuid4().hex[:10]}"
    _seed_compression_chain(db, root, tip)
    entry = store.get_or_create_session(source)
    store.switch_session(entry.session_key, tip)
    return root, tip, entry.session_key


def _bind_raw(store: SessionStore, source, session_id: str) -> str:
    entry = store.get_or_create_session(source)
    store.switch_session(entry.session_key, session_id)
    return entry.session_key


def _make_runner(platform: Platform, *, store: SessionStore,
                 db: SessionDB):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = store
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    # Base R3A: el dispatch toca el reloj scale-to-zero y la resolución
    # canónica usa el SessionDB real del perfil.
    runner._scale_to_zero_note_real_inbound = MagicMock()
    runner._session_db = db
    return runner, adapter


@pytest.fixture(autouse=True)
def _isolate_capture_state():
    reset_active_captures()
    yield
    reset_active_captures()


def test_capture_start_detection():
    assert is_capture_start("captura decidí comprar") == "decidí comprar"
    assert is_capture_start("CAPTURA: algo") == "algo"
    assert is_capture_start("hola mundo") is None
    assert is_capture_start("capturame") is None


def test_bridge_config_requires_explicit_complete_config(monkeypatch):
    _disable_bridge(monkeypatch)
    assert capture_bridge_config() is None
    monkeypatch.setenv("HERMES_CAPTURE_MEMORY_CORE_PYTHON", "python")
    assert capture_bridge_config() is None  # incompleta → deshabilitada
    _enable_bridge(monkeypatch)
    config = capture_bridge_config()
    assert config == {"python": "python", "cwd": "cwd",
                      "profile": "capture-test"}


@pytest.mark.asyncio
async def test_authorized_capture_start_routes_and_skips_agent(
        monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    calls = {}

    async def fake_run(config, argv, env):
        calls["argv"] = argv
        calls["env"] = env
        return 0, json.dumps({"status": "ok", "kind": "clarification",
                              "reply": "¿dominio?", "reason_codes": [],
                              "canonical_confirmed": False, "replay": False}), ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí comprar el auto")
    root, tip, _key = _bind_chain(store, db, event.source)
    reply = await runner._handle_message(event)
    assert reply == "¿dominio?"
    assert agent_calls["n"] == 0  # no llega al agente
    # Identidad canónica R3A: RAÍZ del linaje de compresión (estable tras
    # rotación), JAMÁS la clave de routing ni el session_id físico (TIP).
    routing_key = runner._session_key_for_source(event.source)
    conv_env = calls["env"]["HERMES_CONVERSATION_ID"]
    assert conv_env == root
    assert conv_env != routing_key
    assert conv_env != tip
    assert "decidí" not in conv_env
    argv = calls["argv"]
    assert "--channel" in argv and "telegram" in argv
    assert "--message-id" in argv and "m1" in argv
    assert "--user" in argv and "12345" in argv
    assert "--profile" in argv and "capture-test" in argv
    assert argv[argv.index("--message") + 1] == \
        "captura decidí comprar el auto"


@pytest.mark.asyncio
async def test_identity_separated_after_reset(monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    envs = []

    async def fake_run(config, argv, env):
        envs.append(env["HERMES_CONVERSATION_ID"])
        return 0, json.dumps({"status": "ok", "reply": "ok",
                              "reason_codes": []}), ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = AsyncMock(return_value="agent")

    event_a = _make_event("captura decidí algo")
    root, _tip, key = _bind_chain(store, db, event_a.source)
    await runner._handle_message(event_a)
    assert envs == [root]

    # Reset real: sesión nueva sin padre → linaje propio (separado).
    fresh = f"fresh-{uuid.uuid4().hex[:10]}"
    db.create_session(fresh, source="telegram")
    store.switch_session(key, fresh)
    await runner._handle_message(_make_event("captura decidí otra cosa"))
    assert envs[-1] == fresh
    assert envs[-1] != root


@pytest.mark.asyncio
async def test_identity_separated_after_fork(monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    envs = []

    async def fake_run(config, argv, env):
        envs.append(env["HERMES_CONVERSATION_ID"])
        return 0, json.dumps({"status": "ok", "reply": "ok",
                              "reason_codes": []}), ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = AsyncMock(return_value="agent")

    event_a = _make_event("captura decidí algo")
    root, _tip, key = _bind_chain(store, db, event_a.source)
    await runner._handle_message(event_a)
    assert envs == [root]

    # Fork real (branch): nueva conversación con marcador explícito → el
    # contrato de linaje R3A la separa (raíz propia).
    fork_id = f"fork-{uuid.uuid4().hex[:10]}"
    db.create_session(
        fork_id, source="telegram", parent_session_id=root,
        model_config=json.dumps({"_branched_from": root}))
    assert db.get_compression_lineage(fork_id) == [fork_id]
    store.switch_session(key, fork_id)
    await runner._handle_message(_make_event("captura decidí otra cosa"))
    assert envs[-1] == fork_id
    assert envs[-1] != root


@pytest.mark.asyncio
async def test_explicit_capture_without_mapping_refuses_and_creates_no_session(
        monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    key = runner._session_key_for_source(event.source)
    reply = await runner._handle_message(event)
    assert reply == CaptureIdentityUnavailable.user_message
    assert bridge_calls["n"] == 0  # sin subproceso adapter
    assert agent_calls["n"] == 0  # sin doble ruta con efectos
    # El gate NO crea sesión para satisfacerse.
    assert store.lookup_by_session_key(key) is None


@pytest.mark.asyncio
async def test_explicit_capture_with_stale_mapping_refuses(
        monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    # Mapping a un session_id SIN fila en el SessionDB (stale).
    _bind_raw(store, event.source, f"stale-{uuid.uuid4().hex[:10]}")
    reply = await runner._handle_message(event)
    assert reply == CaptureIdentityUnavailable.user_message
    assert bridge_calls["n"] == 0
    assert agent_calls["n"] == 0


@pytest.mark.asyncio
async def test_ordinary_message_without_identity_keeps_dispatch(
        monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("mensaje ordinario")
    reply = await runner._handle_message(event)
    assert reply == "agent"
    assert agent_calls["n"] == 1
    assert bridge_calls["n"] == 0  # sin identidad canónica no hay ruta


@pytest.mark.asyncio
async def test_off_flow_preserves_ordinary_dispatch(monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, json.dumps({"status": "off_flow", "kind": None,
                              "reply": None, "reason_codes": [],
                              "canonical_confirmed": False,
                              "replay": False}), ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "respuesta ordinaria"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001
    event = _make_event("mensaje ordinario")
    root, _tip, _key = _bind_chain(store, db, event.source)
    # Conversación ACTIVA (hubo un turno de captura ok antes), marcada por
    # la identidad canónica.
    from gateway import capture_bridge
    capture_bridge._ACTIVE_CAPTURES.add(root)

    reply = await runner._handle_message(event)
    assert bridge_calls["n"] == 1  # se consultó al adapter
    assert reply == "respuesta ordinaria"  # off_flow → camino ordinario
    assert agent_calls["n"] == 1


@pytest.mark.asyncio
async def test_unauthorized_never_reaches_bridge(monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = AsyncMock(return_value="agent")

    event = _make_event("captura decidí algo")
    reply = await runner._handle_message(event)
    assert bridge_calls["n"] == 0
    assert reply is None  # remitente no autorizado: silencio/pairing


@pytest.mark.asyncio
async def test_bridge_failure_returns_sanitized_without_agent(
        monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)

    async def fake_run(config, argv, env):
        raise CaptureBridgeFailure("capture_internal_error")

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    _bind_chain(store, db, event.source)
    reply = await runner._handle_message(event)
    assert reply == CaptureBridgeFailure.user_message
    assert agent_calls["n"] == 0  # sin doble ruta con efectos


@pytest.mark.asyncio
async def test_disabled_bridge_keeps_ordinary_dispatch(monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _disable_bridge(monkeypatch)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    reply = await runner._handle_message(event)
    assert reply == "agent"
    assert agent_calls["n"] == 1


@pytest.mark.asyncio
async def test_command_messages_are_not_routed(monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    # /new es comando: no debe pasar por el puente de captura.
    event = _make_event("/new")
    await runner._handle_message(event)
    assert bridge_calls["n"] == 0
    # `/new` es destructivo en esta base: deja un slash-confirm PENDIENTE en
    # el registro global; se cancela para no contaminar el resto de tests
    # (el puente de captura nunca lo consumió).
    from tools import slash_confirm
    key = runner._session_key_for_source(event.source)
    pending = slash_confirm.get_pending(key)
    if pending:
        await slash_confirm.resolve(key, pending.get("confirm_id"), "cancel")


@pytest.mark.asyncio
async def test_pending_slash_confirm_not_consumed_as_capture(
        monkeypatch, real_env):
    """Con un slash-confirm pendiente (trabajo en curso), un mensaje de
    captura NO se consume como captura."""
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    root, _tip, key = _bind_chain(store, db, event.source)
    from gateway import capture_bridge
    capture_bridge._ACTIVE_CAPTURES.add(root)

    # Arma un confirm pendiente REAL (mismo camino que el dispatch: /new).
    await runner._handle_message(_make_event("/new"))
    from tools import slash_confirm
    pending = slash_confirm.get_pending(key)
    assert pending  # el confirm quedó pendiente (trabajo en curso)

    reply = await runner._handle_message(event)
    assert reply == "agent"
    assert agent_calls["n"] == 1
    assert bridge_calls["n"] == 0  # no consumido como captura

    await slash_confirm.resolve(key, pending.get("confirm_id"), "cancel")


@pytest.mark.asyncio
async def test_routed_capture_without_message_id_refuses(
        monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo", message_id=None)
    _bind_chain(store, db, event.source)
    reply = await runner._handle_message(event)
    # Mensaje YA reconocido como captura sin ID estable: rechazo
    # sanitizado; no se fabrica ID ni se deriva al agente.
    assert reply == CaptureMessageIdentityUnavailable.user_message
    assert bridge_calls["n"] == 0
    assert agent_calls["n"] == 0


@pytest.mark.asyncio
async def test_active_capture_without_message_id_refuses(
        monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("mensaje de continuación", message_id=None)
    root, _tip, _key = _bind_chain(store, db, event.source)
    from gateway import capture_bridge
    capture_bridge._ACTIVE_CAPTURES.add(root)
    reply = await runner._handle_message(event)
    assert reply == CaptureMessageIdentityUnavailable.user_message
    assert bridge_calls["n"] == 0
    assert agent_calls["n"] == 0


@pytest.mark.asyncio
async def test_pause_blocks_new_capture(monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    # Pausa global activa: el gate existente responde el aviso de pausa.
    monkeypatch.setattr(
        "agent.estop.paused_reply", lambda: "⏸ Pausado (estop)")
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    _bind_chain(store, db, event.source)
    reply = await runner._handle_message(event)
    assert reply == "⏸ Pausado (estop)"
    assert bridge_calls["n"] == 0  # cero invocaciones del puente en pausa
    assert agent_calls["n"] == 0  # y cero turno de agente


@pytest.mark.asyncio
async def test_inflight_reply_not_consumed_as_capture(monkeypatch, real_env):
    """Un reply de trabajo en curso NUNCA se consume como captura, aunque la
    conversación tenga captura ACTIVA."""
    store, db = real_env
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)

    event = _make_event("mensaje ordinario")
    root, _tip, key = _bind_chain(store, db, event.source)
    from gateway import capture_bridge
    capture_bridge._ACTIVE_CAPTURES.add(root)  # sin el guard se consumiría
    runner._running_agents[key] = object()  # agente en marcha (en curso)

    result = await capture_bridge.handle_capture_message(
        runner, source=event.source, text=event.text, message_id="m9")
    assert result is None
    assert bridge_calls["n"] == 0


@pytest.mark.asyncio
async def test_launch_failure_sanitized_without_raw_details(
        monkeypatch, real_env, caplog, tmp_path):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bogus_python = str(tmp_path / "no-such-python-abc" / "python.exe")
    monkeypatch.setenv("HERMES_CAPTURE_MEMORY_CORE_PYTHON", bogus_python)
    monkeypatch.setenv("HERMES_CAPTURE_MEMORY_CORE_CWD", str(tmp_path))
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    _bind_chain(store, db, event.source)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        reply = await runner._handle_message(event)
    assert reply == CaptureBridgeFailure.user_message
    assert agent_calls["n"] == 0
    # Códigos CONTROLADOS: ni la ruta cruda ni payloads en log/respuesta.
    assert "capture_launch_failed" in caplog.text
    assert bogus_python not in caplog.text
    assert bogus_python not in (reply or "")


@pytest.mark.asyncio
async def test_launch_failure_bad_cwd_controlled_code(tmp_path):
    from gateway import capture_bridge

    with pytest.raises(CaptureBridgeFailure) as excinfo:
        await capture_bridge._run_memory_core(
            {"cwd": str(tmp_path / "no-such-cwd-xyz")},
            [sys.executable, "-c", "print(1)"],
            os.environ.copy(),
        )
    assert str(excinfo.value) == "capture_launch_failed"


@pytest.mark.asyncio
async def test_timeout_controlled_code(monkeypatch, tmp_path):
    from gateway import capture_bridge

    monkeypatch.setattr(capture_bridge, "_DEFAULT_TIMEOUT_S", 0.3)
    with pytest.raises(CaptureBridgeFailure) as excinfo:
        await capture_bridge._run_memory_core(
            {"cwd": str(tmp_path)},
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
        )
    assert str(excinfo.value) == "capture_timeout"


BAD_PROTOCOL_PAYLOADS = [
    ("SENSITIVE!not-json", 0, "capture_bad_json"),
    (json.dumps(["not", "an", "object"]), 0, "capture_bad_json_type"),
    (json.dumps({"status": 7}), 0, "capture_bad_status"),
    (json.dumps({"status": "totally-unknown"}), 0, "capture_bad_status"),
    (json.dumps({"status": "ok", "reply": 7}), 0, "capture_bad_reply"),
    (json.dumps({"status": "ok", "reply": "x"}), 1, "capture_rc_conflict"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("out,rc,expected_code", BAD_PROTOCOL_PAYLOADS)
async def test_protocol_payloads_rejected_sanitized(
        monkeypatch, real_env, caplog, out, rc, expected_code):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)

    async def fake_run(config, argv, env):
        # stderr CRUDO con marcador sensible: nunca debe filtrarse.
        return rc, out, "RAW-STDERR-SENSITIVE"

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    _bind_chain(store, db, event.source)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        reply = await runner._handle_message(event)
    assert reply == CaptureBridgeFailure.user_message
    assert agent_calls["n"] == 0
    assert expected_code in caplog.text
    assert "SENSITIVE" not in caplog.text
    assert "SENSITIVE" not in (reply or "")


# ── E2E real (opt-in): dispatch → adapter memory-core real → base TEST ─────

@pytest.mark.skipif(
    os.environ.get("HERMES_CAPTURE_E2E") != "1",
    reason="E2E de captura requiere HERMES_CAPTURE_E2E=1 y base TEST",
)
@pytest.mark.asyncio
async def test_e2e_dispatch_to_memory_core_adapter(
        monkeypatch, real_env):
    store, db = real_env
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    # La configuración REAL del puente viene del entorno (perfil TEST
    # explícito); el test NO la sustituye. El perfil TEST se materializa en
    # el HERMES_HOME aislado del test (nunca el perfil REAL).
    if capture_bridge_config() is None:
        pytest.skip("E2E requiere configuración real del puente por entorno")
    hermes_home = os.environ.get("HERMES_HOME")
    db_url = os.environ.get("MEMORY_CORE_DB_URL")
    if not hermes_home or not db_url:
        pytest.skip("E2E requiere HERMES_HOME y MEMORY_CORE_DB_URL TEST")
    from pathlib import Path

    profiles = Path(hermes_home) / "memory-core" / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / f"{capture_bridge_config()['profile']}.env").write_text(
        f"MEMORY_CORE_DB_URL={db_url}\n", encoding="utf-8")
    # Conversación ÚNICA por corrida: transporte TEST aislado (sin reutilizar
    # contexto/lineage de corridas anteriores).
    run_id = uuid.uuid4().hex[:12]
    test_user = f"e2e-{run_id}"
    test_chat = f"e2e-chat-{run_id}"
    agent_calls = {"n": 0}

    def _e2e_event(text, message_id):
        return _make_event(text, user_id=test_user, chat_id=test_chat,
                           message_id=message_id)

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM, store=store, db=db)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    # Identidad REAL: cadena de compresión en el SessionDB TEST + mapping del
    # store apuntando al tip físico; el dispatch debe resolver la RAÍZ.
    first = _e2e_event(
        "captura decidí visitar la planta. Dominio work."
        " Las opciones son Visitar en persona y Llamar por"
        " teléfono. Elegí Visitar en persona. Criterio: costo."
        " Decidió Dionisio Vásquez.", "m1")
    root, _tip, _key = _bind_chain(store, db, first.source)

    # inicio (clasificación faltante) → aclaración; nada del agente
    reply = await runner._handle_message(first)
    assert "sensibilidad" in (reply or "").lower()
    assert agent_calls["n"] == 0

    # reenvío CON clasificación (continuación de la conversación activa,
    # SW sin prefijo) → propuesta revisable
    reply = await runner._handle_message(
        _e2e_event("Decidí visitar la planta. Dominio work, sensibilidad"
                   " normal. Las opciones son Visitar en persona y Llamar"
                   " por teléfono. Elegí Visitar en persona. Criterio:"
                   " costo. Área work. Decidió Dionisio Vásquez."
                   " Hecho: stated_rationale: Costo estimado 500 USD."
                   " Vincula Visitar en persona -> stated_rationale.",
                   "m2"))
    assert "confirmo" in (reply or "").lower()
    assert "hechos declarados" in (reply or "")
    assert agent_calls["n"] == 0

    # corrección → nueva versión
    reply = await runner._handle_message(
        _e2e_event("corrige: elegí Llamar por teléfono", "m3"))
    assert "Llamar" in (reply or "")

    # reentrega del MISMO mensaje (m3) → replay sin efectos nuevos
    reply_dup = await runner._handle_message(
        _e2e_event("corrige: elegí Llamar por teléfono", "m3"))
    assert reply_dup == reply

    # confirmación explícita → puente canónico en memory-core
    reply = await runner._handle_message(
        _e2e_event("confirmo", "m4"))
    assert "Confirmación" in (reply or "")

    # mensaje ordinario tras confirmar: off_flow → camino ordinario
    reply = await runner._handle_message(
        _e2e_event("¿qué clima hace?", "m5"))
    assert reply == "agent"
    assert agent_calls["n"] == 1
