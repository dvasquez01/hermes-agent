"""Captura conversacional desde el dispatch autenticado (puente a memory-core).

Ruta: mensaje AUTENTICADO → POST-AUTH y PRE-AGENTE → si es ruta de captura
(``captura …`` explícito o conversación con captura ACTIVA) se invoca la
entrada CLI del adapter de captura de memory-core
(``capture-conversation inbound``) como subproceso con la identidad
CONFIABLE del runtime (``HERMES_CONVERSATION_ID`` derivada del evento
autenticado — jamás del texto del usuario) y su respuesta se devuelve por la
superficie normal del adapter.

Configuración EXPLÍCITA (sin defaults REAL): el puente solo se activa con la
configuración completa por entorno:

* ``HERMES_CAPTURE_MEMORY_CORE_PYTHON`` — intérprete con memory-core;
* ``HERMES_CAPTURE_MEMORY_CORE_CWD``    — raíz del paquete memory-core;
* ``HERMES_CAPTURE_PROFILE``            — perfil TEST explícito.

Opcional: ``MEMORY_CORE_DB_URL`` (base TEST). Sin esta configuración el
puente queda DESHABILITADO y el dispatch continúa ordinario.

Errores: si el puente fue invocado y falla, se devuelve una respuesta
SANITIZADA y el mensaje NO se reenvía al agente (evita doble ruta con
efectos). Si memory-core responde ``off_flow``, el dispatch CONTINÚA
ordinario (cero escrituras de captura).

Marcador de captura activa: por conversación, EN MEMORIA del proceso del
gateway (documentado); tras un reinicio la conversación se retoma con
``captura …``.
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

_ACTIVE_CAPTURES: set[str] = set()


class CaptureBridgeFailure(RuntimeError):
    """Fallo del puente de captura (respuesta sanitizada al usuario)."""

    user_message = (
        "No pude procesar tu mensaje de captura en este momento. "
        "Vuelve a intentarlo en breve.")


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


def _build_env(config: dict, conversation_id: str) -> dict:
    """Entorno del subproceso: identidad CONFIABLE de conversación (nunca del
    texto del usuario), per-per-invocación (sin mutar el entorno del
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
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=config["cwd"], env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out_b, err_b = await asyncio.wait_for(
            process.communicate(), timeout=_DEFAULT_TIMEOUT_S)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        raise CaptureBridgeFailure("timeout") from None
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

    None = deshabilitado / no enrutado / off_flow (cero efectos de captura).
    CaptureBridgeFailure = el puente fue invocado y falló (respuesta
    sanitizada; el llamador NO debe reenviar el mensaje al agente).
    """
    config = capture_bridge_config()
    if config is None:
        return None
    start_turn = is_capture_start(text)
    conversation_id = gateway._session_key_for_source(source)
    routed = start_turn is not None or conversation_id in _ACTIVE_CAPTURES
    if not routed:
        return None
    if not message_id:
        # Sin identidad estable de mensaje no hay replay-dedupe posible:
        # fail-closed al camino ordinario (no se arriesga doble efecto).
        logger.warning("capture bridge: event sin message_id; no enrutado")
        return None

    channel = source.platform.value if source.platform else "unknown"
    user_identity = str(source.user_id)
    env = _build_env(config, conversation_id)
    argv = _build_argv(config, channel=channel, message_id=str(message_id),
                       user_identity=user_identity, text=text)
    rc, out, err = await _run_memory_core(config, argv, env)
    try:
        payload = json.loads(out)
    except ValueError:
        raise CaptureBridgeFailure(
            f"salida no JSON rc={rc} err={err.strip()[:200]}") from None
    status = payload.get("status")
    if status == "off_flow":
        # Mensaje ajeno dentro de conversación con captura activa: el
        # dispatch continúa ordinario SIN efectos; el marcador de captura
        # NO se descarta (la sesión de captura sigue vigente).
        return None
    if status == "ok" and rc == 0:
        _ACTIVE_CAPTURES.add(conversation_id)
        reply = payload.get("reply") or "Listo."
        return reply
    raise CaptureBridgeFailure(
        f"adapter status={status} rc={rc} reasons={payload.get('reason_codes')}")


def maybe_sanitized_failure_message(failure: Exception) -> str:
    return getattr(failure, "user_message", CaptureBridgeFailure.user_message)
