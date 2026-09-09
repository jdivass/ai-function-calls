# Hoja de Trabajo 4 - Function calls & Base de Datos Vectorial

Sistema de preguntas frecuentes (FAQs) con base de datos vectorial PostgreSQL + pgvector y capacidades de Function Calling para un agente conversacional de Parachute S.A.

---

## 1. Infraestructura PostgreSQL + pgvector

La solución utiliza PostgreSQL 16 con la extensión `pgvector`, ejecutado en Docker Compose. Los embeddings son generados con el modelo `all-MiniLM-L6-v2` (384 dimensiones) y se almacenan en la tabla `faq_embeddings` utilizando un índice HNSW para similitud coseno.

### Requisitos

- Docker Engine y Docker Compose v2 (`docker compose`)
- Python 3.10 o superior

### Configuración inicial del entorno

1. Clone el repositorio y sitúese en la raíz del proyecto:
   ```bash
   cd ai-function-calls
   ```

2. Cree su archivo `.env` a partir de la plantilla:
   ```bash
   cp .env.example .env
   ```
   *Nota: Si tiene otro servicio de PostgreSQL corriendo en su máquina en el puerto 5432, puede cambiar `POSTGRES_PORT` en `.env` (por ejemplo a `5434`).*

3. Inicie el contenedor de PostgreSQL con pgvector:
   ```bash
   docker compose up -d
   docker compose ps
   ```

4. Cree y active el entorno virtual de Python:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

5. Instale las dependencias del proyecto:
   ```bash
   pip install -r requirements.txt
   ```
   *(En Linux con CPU, puede instalar previamente PyTorch CPU para optimizar la descarga: `pip install torch --index-url https://download.pytorch.org/whl/cpu`)*.

El contenedor inicializa automáticamente la extensión `vector`, la tabla `faq_embeddings` y el índice HNSW mediante `docker/init.sql`.

---

## 2. Flujo de Trabajo y Verificación

Siga este flujo paso a paso para verificar la infraestructura, cargar los datos y realizar consultas vectoriales:

### Paso A: Verificar la conexión a la base de datos
Compruebe que la conexión y la tabla existan antes de comenzar:
```bash
python src/database.py --test-connection
```
**Salida esperada:**
```text
Probando conexión con PostgreSQL...
✓ Conexión exitosa. Total de FAQs en base de datos: 120
```

### Paso B: Cargar el corpus de FAQs (Persona 1)
Si la base de datos está vacía (`Total de FAQs: 0`) o requiere recargar el dump oficial:
```bash
python src/load_corpus.py
```
El cargador parsea los 120 registros de `data/Corpus_FAQs_Parachute_SA_2026.txt`, genera sus embeddings normalizados con `all-MiniLM-L6-v2` e inserta los registros sin duplicados (`upsert` por `id`).

### Paso C: Consultar el motor de búsqueda vectorial (Persona 2)
Puede realizar consultas en lenguaje natural directamente desde la terminal con `src/database.py`:

```bash
python src/database.py "Tienen fotos o videos de lo que es el salto en cuestión?"
```

**Salida de ejemplo:**
```text
Consultando: 'Tienen fotos o videos de lo que es el salto en cuestión?' (top_k=3, threshold=0.5)

--- Resultado #1 (Similitud: 0.6529) ---
ID: FAQ-081 | Categoría: Fotografía y Contenido Multimedia
Pregunta: ¿Puedo saltar con mi propia cámara Go-Pro o teléfono celular?
Respuesta: Por regulaciones de la Dirección General de Aeronáutica Civil (DGAC) y la USPA, no se permite el uso de cámaras...

--- Resultado #2 (Similitud: 0.6029) ---
ID: FAQ-029 | Categoría: Requisitos Físicos y Salud
Pregunta: ¿Cuánto tiempo debo esperar tras hacer buceo antes de saltar?
Respuesta: Se requiere un intervalo mínimo de 24 horas entre su última inmersión de buceo...
```

Parámetros opcionales:
- `--top-k <N>`: Cantidad máxima de resultados a retornar (por defecto: `3`).
- `--threshold <X>`: Umbral mínimo de similitud coseno (por defecto: `0.50`).

---

## 3. Suite de Pruebas de Búsqueda Vectorial (`test_search.py`)

Se implementó una suite de pruebas automatizada para validar los 3 escenarios fundamentales:

1. **Consultas exactas:** Consultas copiadas textualmente del corpus oficial (deben coincidir con la FAQ esperada con similitud $\ge 0.65$).
2. **Consultas parafraseadas:** Preguntas formuladas con lenguaje coloquial, variaciones de redacción o sinónimos (deben retornar la FAQ correspondiente con similitud $\ge 0.55$).
3. **Consultas fuera de dominio (OOD):** Preguntas ajenas al paracaidismo (recetas, geografía, mecánica) que deben ser descartadas por el umbral retornando 0 resultados.

### Ejecutar las pruebas automatizadas:
```bash
python src/test_search.py
```
**Resultado:** Ejecuta los 9 casos de prueba y genera un reporte detallado:
```text
================================================================================
 SUITE DE PRUEBAS DE BÚSQUEDA VECTORIAL - PARACHUTE S.A.
 Configuración: threshold=0.50, top_k=3
================================================================================
...
================================================================================
 RESULTADO FINAL: 9/9 pruebas superadas.
================================================================================
```

### Modo interactivo para pruebas libres:
Permite ingresar consultas en tiempo real por consola para explorar el comportamiento del motor:
```bash
python src/test_search.py --interactive
```

---

## 4. Integración con el Agente (Persona 3)

El módulo `src/database.py` expone la función oficial para la herramienta / tool del agente:

```python
from database import search_knowledge_base

# Llamada típica desde el tool del LLM
resultados = search_knowledge_base(query="¿Cuál es el peso máximo?", top_k=3)
```

### Formato retornado
Retorna una lista de diccionarios con las FAQs más relevantes:
```json
[
  {
    "id": "FAQ-021",
    "categoria": "Requisitos Físicos y Salud",
    "pregunta": "¿Cuál es el peso máximo permitido para saltar?",
    "respuesta": "El límite de peso máximo estricto para realizar el salto tándem es de 100 kg...",
    "metadata": {
      "fecha": "2026-09-29",
      "empresa": "Parachute S.A.",
      "unidad_medida": "Sistema Métrico Decimal"
    },
    "similarity": 0.7491
  }
]
```

### Manejo de preguntas no respaldadas
Si la consulta del usuario no tiene relación con el evento o su similitud está por debajo del umbral (`threshold`), la función retorna una lista vacía `[]`.
En ese caso, las instrucciones del agente LLM deben indicarle admitir amablemente que no dispone de dicha información en su base de conocimientos.

---

## 5. Estructura del Proyecto

```text
ai-function-calls/
├── data/
│   └── Corpus_FAQs_Parachute_SA_2026.txt  # Dump oficial de 120 FAQs
├── docker/
│   └── init.sql                          # Esquema de BD, extensión pgvector e índice HNSW
├── src/
│   ├── database.py                       # Conexión a PostgreSQL y motor de búsqueda vectorial
│   ├── load_corpus.py                    # Parser del TXT y cargador con embeddings
│   └── test_search.py                    # Suite de pruebas automatizadas y CLI interactivo
├── docker-compose.yml                    # Definición del contenedor PostgreSQL + pgvector
├── requirements.txt                      # Dependencias del proyecto
├── .env.example                          # Plantilla de variables de entorno
└── README.md                             # Documentación del proyecto
```

---

## 6. Reinicializar la Base de Datos

Para borrar los datos persistidos y recrear el esquema desde cero:

```bash
docker compose down -v
docker compose up -d
python src/load_corpus.py
```
