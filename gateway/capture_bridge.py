"""Captura conversacional desde el dispatch autenticado (puente a memory-core).

Ruta: mensaje AUTENTICADO → POST-AUTH, POST-PAUSA y POST-CONTROLES de
trabajo en curso → si es ruta de captura (``captura …`` explícito o
conversación con captura ACTIVA) se invoca la entrada CLI del adapter de
captura de memory-core (``capture-conversation inbound``) como subproceso
con la identidad CONFIABLE del runtime — la identidad LÓGICA canónica R3A
(``HERMES_CONVERSATION_ID``; jamás derivada del texto del usuario) — y su
respuesta se devuelve por la superficie normal del adapter.

Semántica de identidad (R3A, estricta): la identidad se resuelve por la
cadena REAL del runtime, sin inventar sesiones y sin fallback:

    session_key (routing)
        → SessionStore.peek_session_id(session_key)   [solo lectura]
        → resolve_conversation_identity(session_id, SessionDB del perfil)

Sin mapping, fila o linaje válido: la ruta explícita de captura se rechaza
con respuesta sanitizada (sin agente y sin subproceso) y los mensajes
ordinarios siguen el dispatch normal. La clave de routing JAMÁS se usa como
session_id ni como identidad canónica.

Trabajo en curso: los comandos, las respuestas a prompts de trabajo en curso
(update prompt, clarify, slash-confirm, aprobaciones) y el steering de un
agente en marcha NUNCA se consumen como captura.

Configuración EXPLÍCITA (sin defaults REAL): el puente solo se activa con la
configuración completa por entorno:

* ``HERMES_CAPTURE_MEMORY_CORE_PYTHON`` — intérprete con memory-core;
* ``HERMES_CAPTURE_MEMORY_CORE_CWD``    — raíz del paquete memory-core;
* ``HERMES_CAPTURE_PROFILE``            — perfil TEST explícito.

Opcional: ``MEMORY_CORE_DB_URL`` (base TEST). Sin esta configuración el
puente queda DESHABILITADO y el dispatch continúa ordinario.

Errores: si el puente fue invocado y falla (lanzamiento, timeout, protocolo
JSON inadmisible), se devuelve una respuesta SANITIZADA y el mensaje NO se
reenvía al agente (evita doble ruta con efectos). Si memory-core responde
``off_flow``, el dispatch CONTINÚA ordinario (cero escrituras de captura).
Los fallos usan SOLO códigos controlados: nunca stderr/stdout crudos, texto
del usuario, DSN ni payloads.

Marcador de captura activa: por identidad canónica de conversación, EN
MEMORIA del proceso del gateway (documentado); tras un reinicio la
conversación se retoma con ``captura …``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

CAPTURE_PREFIXES = ("captura ", "captura:")
_DEFAULT_TIMEOUT_S = 60.0

_OK_STATUSES = ("ok", "off_flow")

_ACTIVE_CAPTURES: set[str] = set()


class CaptureBridgeFailure(RuntimeError):
    """Fallo del puente de captura (respuesta sanitizada al usuario).

    ``str(exc)`` es un código CONTROLADO apto para logs: nunca contiene
    stderr/stdout, texto del usuario, DSN ni payloads.
    """

    user_message = (
        "No pude procesar tu mensaje de captura en este momento. "
        "Vuelve a intentarlo en breve.")


class CaptureIdentityUnavailable(CaptureBridgeFailure):
    """Ruta de captura sin identidad canónica de conversación (fail-closed)."""

    user_message = (
        "No puedo asociar esta conversación a una identidad canónica "
        "todavía. Envía un mensaje normal y reintenta la captura después.")


class CaptureMessageIdentityUnavailable(CaptureBridgeFailure):
    """Mensaje de captura sin identidad estable de mensaje (fail-closed)."""

    user_message = (
        "No pude verificar el identificador del mensaje de captura. "
        "Reenvíalo desde la aplicación e inténtalo de nuevo.")


def capture_bridge_config() -> Optional[dict]:
    """Configuración EXPLÍCITA del puente (None → deshabilitado)."""
    python = os.environ.get("HERMES_CAPTURE_MEMORY_CORE_PYTHON", "").strip()
    cwd = os.environ.get("HERMES_CAPTURE_MEMORY_CORE_CWD", "").strip()
    profile = os.environ.get("HERMES_CAPTURE_PROFILE", "").strip()
    if not python or not cwd or not profile:
        return None
    config = {"python": python, "cwd": cwd, "profile": profile}
    db_url = os.environ.get("MEMORY_CORE_DB_URL", "").strip()
    if db_url:
        config["db_url"] = db_url
    return config


def is_capture_start(text: str) -> Optional[str]:
    """Ruta EXPLÍCITA de inicio: devuelve el turno de captura o None."""
    stripped = (text or "").strip()
    lowered = stripped.lower()
    for prefix in CAPTURE_PREFIXES:
        if lowered.startswith(prefix):
            if len(stripped) <= len(prefix):
                return ""
            return stripped[len(prefix):].strip()
    return None


def reset_active_captures() -> None:
    """Higiene de tests."""
    _ACTIVE_CAPTURES.clear()


def _resolve_canonical_conversation_id(gateway, source) -> str:
    """Identidad LÓGICA canónica del turno por la cadena REAL del runtime.

    ``session_key`` (routing) → ``SessionStore.peek_session_id`` (solo
    lectura: JAMÁS crea sesión) → ``resolve_conversation_identity`` sobre el
    SessionDB del perfil (estricta: "" si no hay row/linaje válido; la clave
    de routing no se usa como fallback ni como session_id).
    """
    try:
        from agent.conversation_identity import resolve_conversation_identity
    except Exception:  # pragma: no cover — import boundary defensivo
        return ""
    try:
        session_key = gateway._session_key_for_source(source)
    except Exception:
        logger.debug("capture bridge: session key resolution failed")
        return ""
    if not session_key:
        return ""
    session_id = ""
    try:
        store = getattr(gateway, "session_store", None)
        peek = getattr(store, "peek_session_id", None)
        if callable(peek):
            session_id = str(peek(session_key) or "")
    except Exception:
        logger.debug("capture bridge: session id lookup failed")
        return ""
    if not session_id:
        return ""
    session_db = getattr(gateway, "_session_db", None)
    session_db = getattr(session_db, "_db", session_db)
    return resolve_conversation_identity(session_id, session_db) or ""


def _inflight_blocks_capture(gateway, session_key: str) -> bool:
    """True cuando la sesión tiene trabajo en curso: su reply NUNCA se
    consume como captura (agente en marcha, clarify, slash-confirm,
    aprobación de tool o update prompt pendientes)."""
    if not session_key:
        return False
    try:
        if gateway._is_session_running(session_key):
            return True
    except Exception:
        pass
    try:
        state = gateway._peek_session_state(session_key)
        if state is not None and state.persistent.update_prompt_pending:
            return True
    except Exception:
        pass
    try:
        from tools import clarify_gateway
        if clarify_gateway.get_pending_for_session(
                session_key, include_choice_prompts=True):
            return True
    except Exception:
        pass
    try:
        from tools import slash_confirm
        if slash_confirm.get_pending(session_key):
            return True
    except Exception:
        pass
    try:
        from tools.approval import has_blocking_approval
        if has_blocking_approval(session_key):
            return True
    except Exception:
        pass
    return False


def _build_env(config: dict, conversation_id: str) -> dict:
    """Entorno del subproceso: identidad canónica CONFIABLE (jamás derivada
    del texto del usuario), por-invocación (sin mutar el entorno del
    proceso)."""
    try:
        from tools.environments.local import hermes_subprocess_env

        env = hermes_subprocess_env()
    except Exception:  # pragma: no cover — import boundary defensivo
        env = os.environ.copy()
    env["HERMES_CONVERSATION_ID"] = conversation_id
    if config.get("db_url"):
        env["MEMORY_CORE_DB_URL"] = config["db_url"]
    return env


def _build_argv(config: dict, *, channel: str, message_id: str,
                user_identity: str, text: str) -> list[str]:
    return [
        config["python"], "-m", "hermes_memory_core.cli",
        "capture-conversation", "inbound",
        "--channel", channel,
        "--message-id", message_id,
        "--user", user_identity,
        "--message", text,
        "--profile", config["profile"],
        "--json",
    ]


async def _run_memory_core(config: dict, argv: list[str],
                           env: dict) -> tuple[int, str, str]:
    """Ejecuta el adapter como subproceso.

    Los fallos de LANZAMIENTO (intérprete/cwd inválidos) y los TIMEOUT se
    convierten en ``CaptureBridgeFailure`` con código CONTROLADO, dentro de
    esta frontera: el llamador solo ve la respuesta sanitizada.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=config["cwd"], env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except Exception:
        raise CaptureBridgeFailure("capture_launch_failed") from None
    try:
        out_b, err_b = await asyncio.wait_for(
            process.communicate(), timeout=_DEFAULT_TIMEOUT_S)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except Exception:
            pass
        try:
            await process.communicate()
        except Exception:
            pass
        raise CaptureBridgeFailure("capture_timeout") from None
    return process.returncode or 0, out_b.decode("utf-8", "replace"), \
        err_b.decode("utf-8", "replace")


async def handle_capture_message(
    gateway,
    *,
    source,
    text: str,
    message_id: Optional[str],
) -> Optional[str]:
    """Devuelve la respuesta del adapter o None si el dispatch continúa.

    None = deshabilitado / no enrutado / off_flow / trabajo en curso (cero
    efectos de captura). CaptureBridgeFailure = el puente fue invocado y
    falló (respuesta sanitizada; el llamador NO debe reenviar el mensaje al
    agente). CaptureIdentityUnavailable / CaptureMessageIdentityUnavailable
    = fronteras fail-closed (misma respuesta sanitizada; sin subproceso y
    sin agente).
    """
    config = capture_bridge_config()
    if config is None:
        return None
    start_turn = is_capture_start(text)
    if start_turn is None:
        # Un mensaje ordinario solo puede enrutarse a captura si la
        # conversación tiene una captura ACTIVA; el trabajo en curso nunca
        # se consume como captura.
        try:
            session_key = gateway._session_key_for_source(source)
        except Exception:
            session_key = ""
        if _inflight_blocks_capture(gateway, session_key):
            return None
    conversation_id = _resolve_canonical_conversation_id(gateway, source)
    if not conversation_id:
        if start_turn is not None:
            # Ruta EXPLÍCITA sin identidad canónica: el texto de captura no
            # se reenvía al agente ni se invoca el adapter (fail-closed).
            raise CaptureIdentityUnavailable("capture_identity_unavailable")
        # Sin identidad canónica no hay conversación de captura ACTIVA
        # posible; el dispatch sigue ordinario (cero efectos de captura).
        return None
    routed = start_turn is not None or conversation_id in _ACTIVE_CAPTURES
    if not routed:
        return None
    if not message_id:
        # Mensaje YA reconocido como captura sin identidad estable de
        # mensaje: rechazo sanitizado (sin subproceso y sin agente); no se
        # fabrica un ID ni se deriva el mensaje al agente.
        raise CaptureMessageIdentityUnavailable(
            "capture_message_identity_missing")

    channel = source.platform.value if source.platform else "unknown"
    user_identity = str(source.user_id)
    env = _build_env(config, conversation_id)
    argv = _build_argv(config, channel=channel, message_id=str(message_id),
                       user_identity=user_identity, text=text)
    rc, out, _err = await _run_memory_core(config, argv, env)
    try:
        payload = json.loads(out)
    except ValueError:
        raise CaptureBridgeFailure("capture_bad_json") from None
    if not isinstance(payload, dict):
        raise CaptureBridgeFailure("capture_bad_json_type") from None
    status = payload.get("status")
    if not isinstance(status, str) or status not in _OK_STATUSES:
        raise CaptureBridgeFailure("capture_bad_status") from None
    if status == "off_flow":
        # Mensaje ajeno dentro de conversación con captura activa: el
        # dispatch continúa ordinario SIN efectos; el marcador de captura
        # NO se descarta (la sesión de captura sigue vigente).
        return None
    if rc != 0:
        raise CaptureBridgeFailure("capture_rc_conflict") from None
    reply = payload.get("reply")
    if reply is None:
        reply = "Listo."
    elif not isinstance(reply, str):
        raise CaptureBridgeFailure("capture_bad_reply") from None
    _ACTIVE_CAPTURES.add(conversation_id)
    return reply


def maybe_sanitized_failure_message(failure: Exception) -> str:
    return getattr(failure, "user_message", CaptureBridgeFailure.user_message)
