"""Agente conversacional con function calling para las FAQs de Parachute S.A."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

from groq_client import get_groq_client, get_groq_model
from tools import TOOL_EXECUTORS, TOOLS


SYSTEM_PROMPT = """Eres el agente de preguntas frecuentes de Parachute S.A.

Responde exclusivamente con información contenida en los resultados de
search_knowledge_base. No uses conocimiento externo, memoria ni suposiciones.
Debes utilizar la herramienta antes de responder cada pregunta.
Si la herramienta devuelve una lista vacía o los resultados no contienen
información suficiente, responde:
\"Lo siento, no puedo responder esa pregunta con la información disponible en la
base de conocimientos de Parachute S.A.\"
No inventes datos. Responde en español y de forma clara y concisa.
"""

NO_ANSWER = (
    "Lo siento, no puedo responder esa pregunta con la información disponible en la "
    "base de conocimientos de Parachute S.A."
)
EXIT_COMMANDS = {"bye", "salir", "exit", "quit"}
SESSION_END_MESSAGE = "Sesión finalizada."


def _assistant_message_dict(message: Any) -> dict[str, Any]:
    """Convierte el mensaje del SDK al formato requerido por la siguiente llamada."""
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in (message.tool_calls or [])
        ],
    }


def _execute_tool_call(tool_call: Any) -> str:
    """Ejecuta un tool call y convierte errores en resultados legibles para el LLM."""
    name = tool_call.function.name
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return json.dumps({"error": f"Herramienta no disponible: {name}"})

    try:
        return executor(tool_call.function.arguments)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return json.dumps({"error": f"No se pudo ejecutar la herramienta: {exc}"})


def run_agent_turn(
    client: Any,
    messages: list[dict[str, Any]],
    user_query: str,
    *,
    max_tool_rounds: int = 4,
) -> str:
    """Procesa una pregunta, sus tool calls y devuelve la respuesta final.

    ``messages`` se modifica para conservar el historial de la sesión.
    """
    if not user_query.strip():
        return ""
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds debe ser mayor que cero.")

    messages.append({"role": "user", "content": user_query.strip()})

    for round_number in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=get_groq_model(),
            messages=messages,
            tools=TOOLS,
            tool_choice="required" if round_number == 0 else "auto",
        )
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            answer = (assistant_message.content or "").strip()
            return answer or NO_ANSWER

        messages.append(_assistant_message_dict(assistant_message))
        for tool_call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": _execute_tool_call(tool_call),
                }
            )

    raise RuntimeError("El modelo excedió el máximo de rondas de herramientas.")


def run_interactive_session(
    client: Any,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Mantiene una sesión activa y conserva el historial entre preguntas."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    output_fn("Agente Parachute S.A. listo. Escribe 'Bye' para salir.")

    while True:
        try:
            user_query = input_fn("\nPregunta > ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn(f"\n{SESSION_END_MESSAGE}")
            return 0

        if user_query.lower() in EXIT_COMMANDS:
            output_fn(SESSION_END_MESSAGE)
            return 0
        if not user_query:
            continue

        try:
            answer = run_agent_turn(client, messages, user_query)
            output_fn(f"\nRespuesta: {answer}")
        except KeyboardInterrupt:
            output_fn(f"\n{SESSION_END_MESSAGE}")
            return 0
        except Exception as exc:
            output_fn(f"Error al procesar la pregunta: {exc}")


def main() -> int:
    try:
        client = get_groq_client()
    except ValueError as exc:
        print(f"Error de configuración: {exc}", file=sys.stderr)
        return 1

    return run_interactive_session(client)


if __name__ == "__main__":
    raise SystemExit(main())
