# Guía de Integración para el Agente (Persona 3)

Este documento especifica el contrato de uso de la herramienta de búsqueda vectorial implementada en `src/database.py` para ser consumida por el agente conversacional (LLM / Groq SDK).

---

## 1. Función de Búsqueda Vectorial

El módulo `src/database.py` expone la función principal lista para utilizarse dentro del tool / function call del LLM:

```python
from database import search_knowledge_base

# Uso básico:
resultados = search_knowledge_base(query="¿Cuál es el peso máximo permitido para saltar?")

# Uso con parámetros personalizados:
resultados = search_knowledge_base(
    query="¿Tienen fotos o videos del salto?",
    top_k=3,           # Cantidad máxima de FAQs a recuperar (por defecto: 3)
    threshold=0.50     # Umbral de similitud coseno mínima (por defecto: 0.50 o SIMILARITY_THRESHOLD en .env)
)
```

---

## 2. Contrato de Respuesta

La función retorna una **lista de diccionarios** (`list[dict]`). Cada elemento representa una FAQ relevante ordenada de mayor a menor similitud:

```json
[
  {
    "id": "FAQ-021",
    "categoria": "Requisitos Físicos y Salud",
    "pregunta": "¿Cuál es el peso máximo permitido para saltar?",
    "respuesta": "El límite de peso máximo estricto para realizar el salto tándem es de 100 kg. Cada participante es pesado durante la fase de registro. Si el peso oscila entre 90 kg y 100 kg, se aplicará un recargo administrativo adicional de Q250 por balance de carga de la aeronave.",
    "metadata": {
      "empresa": "Parachute S.A.",
      "evento": "Gran Evento de Paracaidismo Guatemala 2026",
      "fecha": "2026-09-29",
      "unidad_medida": "Sistema Métrico Decimal"
    },
    "similarity": 0.7491
  }
]
```

### Campos devueltos:
- `id`: Identificador único de la FAQ (ej. `FAQ-021`).
- `categoria`: Categoría temática oficial.
- `pregunta`: Pregunta registrada en el corpus.
- `respuesta`: Respuesta oficial validada por Parachute S.A.
- `metadata`: Objeto con metadatos asociados (fecha, empresa, etc.).
- `similarity`: Puntaje de similitud coseno calculado con `pgvector` ($1 - distancia\_coseno$).

---

## 3. Manejo de Preguntas Fuera de Dominio (No respaldadas)

Cuando el usuario formula una pregunta que no tiene relación con el evento de paracaidismo o cuya similitud no alcanza el umbral de corte, la función retorna una **lista vacía**:

```python
resultados = search_knowledge_base("¿Cómo preparar una pizza de pepperoni?")
# resultados == []
```

### Comportamiento esperado del agente:
Siguiendo las instrucciones de la Hoja de Trabajo #4:
> *"Responder únicamente basado en la información del archivo. Si la pregunta que el usuario hace no está en el agente tiene que admitir que no la puede responder."*

Cuando `search_knowledge_base` retorne `[]`, el prompt del agente debe indicarle responder explícitamente algo como:
> *"Lo siento, únicamente dispongo de información sobre el evento de paracaidismo de Parachute S.A. y no cuento con detalles para responder a esa consulta."*

---

## 4. Ejemplo de Definición de Tool para Groq / OpenAI SDK

Para registrar la herramienta en el cliente de Groq/OpenAI compatible, se puede definir el schema de la siguiente manera:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Consulta la base de conocimientos oficial de Parachute S.A. para responder preguntas frecuentes sobre el evento de paracaidismo (horarios, requisitos, ubicación, costos, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La consulta o pregunta específica a buscar en la base de datos de FAQs."
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```
