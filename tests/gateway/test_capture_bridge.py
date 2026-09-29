"""Conexión del dispatch autenticado al adapter de captura (memory-core).

Cubre la ruta POST-AUTH y PRE-AGENTE insertada en GatewayRunner._handle_message:
- ruta explícita "captura …" y continuación de conversación activa;
- off_flow conserva el camino ordinario (sin efectos de captura);
- errores del puente → respuesta sanitizada SIN reenviar al agente;
- autorización ANTES del routing (no autorizado → cero invocaciones);
- identidad de conversación EXCLUSIVAMENTE desde el evento autenticado;
- comandos (p. ej. /new) no se enrutan.

El E2E real (HERMES_CAPTURE_E2E=1) invoca el adapter REAL de memory-core como
subproceso contra la base TEST desechable, a través del MISMO dispatch.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.capture_bridge import (
    CaptureBridgeFailure,
    CaptureIdentityUnavailable,
    capture_bridge_config,
    is_capture_start,
    reset_active_captures,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource

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


class _FakeSessionDB:
    """SessionDB mínimo: solo el walk estricto de linaje (R3A)."""

    def __init__(self, root: Optional[str]):
        self._root = root

    def get_compression_lineage(self, session_id: str):
        if not self._root:
            return []
        return [self._root]


def _make_runner(platform: Platform, *,
                 canonical_id: Optional[str] = "conv-canonical-1"):
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
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    # Base R3A: el dispatch toca el reloj scale-to-zero y la resolución
    # canónica usa el SessionDB del gateway.
    runner._scale_to_zero_note_real_inbound = MagicMock()
    runner._session_db = _FakeSessionDB(canonical_id)
    return runner, adapter


@pytest.fixture(autouse=True)
def _isolate_capture_state(monkeypatch):
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
        monkeypatch):
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

    runner, _adapter = _make_runner(Platform.TELEGRAM)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí comprar el auto")
    reply = await runner._handle_message(event)
    assert reply == "¿dominio?"
    assert agent_calls["n"] == 0  # no llega al agente
    # Identidad CONFIABLE: la identidad canónica R3A resuelta del SessionDB
    # (jamás la clave física de sesión, jamás el texto del usuario).
    assert calls["env"]["HERMES_CONVERSATION_ID"] == "conv-canonical-1"
    assert calls["env"]["HERMES_CONVERSATION_ID"] != \
        runner._session_key_for_source(event.source)
    assert "decidí" not in calls["env"]["HERMES_CONVERSATION_ID"]
    argv = calls["argv"]
    assert "--channel" in argv and "telegram" in argv
    assert "--message-id" in argv and "m1" in argv
    assert "--user" in argv and "12345" in argv
    assert "--profile" in argv and "capture-test" in argv
    assert argv[argv.index("--message") + 1] == \
        "captura decidí comprar el auto"


@pytest.mark.asyncio
async def test_off_flow_preserves_ordinary_dispatch(monkeypatch):
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

    runner, _adapter = _make_runner(Platform.TELEGRAM)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001
    # Conversación ACTIVA (hubo un turno de captura ok antes), marcada por
    # identidad canónica R3A.
    from gateway import capture_bridge
    capture_bridge._ACTIVE_CAPTURES.add("conv-canonical-1")

    event = _make_event("mensaje ordinario")
    reply = await runner._handle_message(event)
    assert bridge_calls["n"] == 1  # se consultó al adapter
    assert reply == "respuesta ordinaria"  # off_flow → camino ordinario
    assert agent_calls["n"] == 1


@pytest.mark.asyncio
async def test_unauthorized_never_reaches_bridge(monkeypatch):
    _clear_auth_env(monkeypatch)
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    runner, _adapter = _make_runner(Platform.TELEGRAM)
    runner._handle_message_with_agent = AsyncMock(return_value="agent")

    event = _make_event("captura decidí algo")
    reply = await runner._handle_message(event)
    assert bridge_calls["n"] == 0
    assert reply is None  # remitente no autorizado: silencio/pairing


@pytest.mark.asyncio
async def test_bridge_failure_returns_sanitized_without_agent(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)

    async def fake_run(config, argv, env):
        raise CaptureBridgeFailure("boom")

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    reply = await runner._handle_message(event)
    assert reply == CaptureBridgeFailure.user_message
    assert agent_calls["n"] == 0  # sin doble ruta con efectos


@pytest.mark.asyncio
async def test_disabled_bridge_keeps_ordinary_dispatch(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _disable_bridge(monkeypatch)
    agent_calls = {"n": 0}

    async def agent_stub(event, source, _quick_key, _run_generation):
        agent_calls["n"] += 1
        return "agent"

    runner, _adapter = _make_runner(Platform.TELEGRAM)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    reply = await runner._handle_message(event)
    assert reply == "agent"
    assert agent_calls["n"] == 1


@pytest.mark.asyncio
async def test_command_messages_are_not_routed(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    _enable_bridge(monkeypatch)
    bridge_calls = {"n": 0}

    async def fake_run(config, argv, env):
        bridge_calls["n"] += 1
        return 0, "{}", ""

    monkeypatch.setattr("gateway.capture_bridge._run_memory_core", fake_run)
    runner, _adapter = _make_runner(Platform.TELEGRAM)
    # /new es comando: no debe pasar por el puente de captura.
    event = _make_event("/new")
    await runner._handle_message(event)
    assert bridge_calls["n"] == 0


@pytest.mark.asyncio
async def test_explicit_capture_without_canonical_identity_refuses_sanitized(
        monkeypatch):
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

    # Sin identidad canónica (p. ej. sesión aún no persistida, /new):
    # fail-closed — el texto de captura NO va al agente ni al subproceso.
    runner, _adapter = _make_runner(Platform.TELEGRAM, canonical_id=None)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("captura decidí algo")
    reply = await runner._handle_message(event)
    assert reply == CaptureIdentityUnavailable.user_message
    assert bridge_calls["n"] == 0  # sin subproceso adapter
    assert agent_calls["n"] == 0  # sin doble ruta con efectos


@pytest.mark.asyncio
async def test_ordinary_message_without_canonical_identity_keeps_dispatch(
        monkeypatch):
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

    runner, _adapter = _make_runner(Platform.TELEGRAM, canonical_id=None)
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    event = _make_event("mensaje ordinario")
    reply = await runner._handle_message(event)
    assert reply == "agent"
    assert agent_calls["n"] == 1
    assert bridge_calls["n"] == 0  # sin identidad canónica no hay ruta


# ── E2E real (opt-in): dispatch → adapter memory-core real → base TEST ─────

@pytest.mark.skipif(
    os.environ.get("HERMES_CAPTURE_E2E") != "1",
    reason="E2E de captura requiere HERMES_CAPTURE_E2E=1 y base TEST",
)
@pytest.mark.asyncio
async def test_e2e_dispatch_to_memory_core_adapter(monkeypatch):
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
    import uuid

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

    runner, _adapter = _make_runner(
        Platform.TELEGRAM, canonical_id=f"e2e-conv-{run_id}")
    runner._handle_message_with_agent = agent_stub  # noqa: SLF001

    # inicio (clasificación faltante) → aclaración; nada del agente
    reply = await runner._handle_message(
        _e2e_event("captura decidí visitar la planta. Dominio work."
                   " Las opciones son Visitar en persona y Llamar por"
                   " teléfono. Elegí Visitar en persona. Criterio: costo."
                   " Decidió Dionisio Vásquez.", "m1"))
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
