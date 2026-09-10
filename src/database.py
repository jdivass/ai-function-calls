"""Módulo de base de datos y búsqueda vectorial para las FAQs de Parachute S.A."""

from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any, Generator

import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# El modelo público puede descargarse sin HF_TOKEN; evita mostrar el aviso
# informativo de autenticación sin ocultar errores reales de Hugging Face.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings(
    "ignore",
    message=r"You are sending unauthenticated requests to the HF Hub.*",
)
from sentence_transformers import SentenceTransformer

# Ruta raíz del proyecto y carga de variables de entorno
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# Variable global para cachear el modelo de embeddings en memoria (Singleton)
_MODEL_INSTANCE: SentenceTransformer | None = None


def get_db_config() -> dict[str, Any]:
    """Obtiene y valida la configuración de conexión a PostgreSQL."""
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise ValueError(
            "Falta definir la variable de entorno POSTGRES_PASSWORD. "
            "Asegúrate de configurar tu archivo .env basado en .env.example."
        )

    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", "parachute_faqs"),
        "user": os.getenv("POSTGRES_USER", "parachute"),
        "password": password,
    }


@contextmanager
def get_connection() -> Generator[PgConnection, None, None]:
    """Context manager para gestionar conexiones a PostgreSQL de forma segura.

    Maneja automáticamente el cierre y rollback en caso de error, y proporciona
    mensajes de diagnóstico claros ante problemas de conexión.
    """
    config = get_db_config()
    conn = None
    try:
        conn = psycopg2.connect(**config)
        yield conn
    except psycopg2.OperationalError as exc:
        raise ConnectionError(
            f"No se pudo conectar a PostgreSQL en {config['host']}:{config['port']}/"
            f"{config['dbname']} con usuario '{config['user']}'. "
            f"Verifica que el contenedor de Docker esté activo ('docker compose up -d').\n"
            f"Detalle técnico: {exc}"
        ) from exc
    finally:
        if conn is not None and not conn.closed:
            conn.close()


def get_embedding_model() -> SentenceTransformer:
    """Carga y reutiliza en memoria el modelo all-MiniLM-L6-v2."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        # transformers muestra una barra "Loading weights" por stderr al
        # inicializar el modelo. La ocultamos solo durante esa carga; los
        # errores siguen propagándose normalmente.
        with StringIO() as suppressed_stderr:
            with redirect_stderr(suppressed_stderr):
                _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME)
    return _MODEL_INSTANCE


def generate_query_embedding(query: str) -> list[float]:
    """Convierte el texto de la consulta en un vector normalizado de 384 dimensiones."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("La consulta no puede estar vacía.")

    model = get_embedding_model()
    vector = model.encode(cleaned_query, normalize_embeddings=True)
    return [float(v) for v in vector]


def search_faqs(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Busca en faq_embeddings las FAQs más cercanas utilizando distancia coseno.

    Args:
        query: Consulta o pregunta formulada por el usuario.
        top_k: Cantidad máxima de resultados a retornar.
        threshold: Similitud mínima (entre 0.0 y 1.0) para considerar un resultado válido.
                   Los resultados con similitud menor al umbral serán descartados.

    Returns:
        Lista de diccionarios con las FAQs relevantes y su puntaje de similitud.
        Si ningún resultado supera el umbral, retorna una lista vacía.
    """
    if not query or not query.strip():
        return []

    # 1. Vectorizar la consulta con el mismo modelo y normalización
    query_vector = generate_query_embedding(query)
    vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    # 2. Consultar PostgreSQL usando el operador <=> de distancia coseno y el índice HNSW
    sql = """
    SELECT
        id,
        categoria,
        pregunta,
        respuesta,
        metadata,
        1 - (embedding <=> %s::vector) AS similarity
    FROM faq_embeddings
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    results: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (vector_str, vector_str, top_k))
            rows = cur.fetchall()

            for row in rows:
                similarity = float(row["similarity"])
                # Filtrar resultados que no alcancen el umbral de similitud
                if similarity >= threshold:
                    results.append(
                        {
                            "id": row["id"],
                            "categoria": row["categoria"],
                            "pregunta": row["pregunta"],
                            "respuesta": row["respuesta"],
                            "metadata": row["metadata"],
                            "similarity": round(similarity, 4),
                        }
                    )

    return results


def search_knowledge_base(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Función de interfaz para la herramienta (Tool/Function Calling) del agente.

    Permite a la Persona 3 invocar directamente la búsqueda sobre la base de
    conocimientos sin preocuparse por la conexión ni la vectorización.
    """
    applied_threshold = (
        threshold if threshold is not None else DEFAULT_SIMILARITY_THRESHOLD
    )
    return search_faqs(query=query, top_k=top_k, threshold=applied_threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Búsqueda vectorial en FAQs de Parachute S.A."
    )
    parser.add_argument("query", nargs="?", help="Pregunta a consultar")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Número máximo de resultados (por defecto: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Umbral mínimo de similitud (por defecto: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Verificar la conexión a PostgreSQL y estado de la tabla",
    )
    args = parser.parse_args()

    if args.test_connection or not args.query:
        print("Probando conexión con PostgreSQL...")
        try:
            with get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM faq_embeddings;")
                    total = cursor.fetchone()[0]
                    print(f"✓ Conexión exitosa. Total de FAQs en base de datos: {total}")
        except Exception as err:
            print(f"✗ Error al conectar: {err}", file=sys.stderr)
            sys.exit(1)

        if not args.query:
            sys.exit(0)

    print(f"\nConsultando: '{args.query}' (top_k={args.top_k}, threshold={args.threshold})")
    matches = search_knowledge_base(
        args.query, top_k=args.top_k, threshold=args.threshold
    )
    if not matches:
        print("No se encontraron FAQs que superen el umbral de similitud.")
    else:
        for idx, match in enumerate(matches, 1):
            print(f"\n--- Resultado #{idx} (Similitud: {match['similarity']:.4f}) ---")
            print(f"ID: {match['id']} | Categoría: {match['categoria']}")
            print(f"Pregunta: {match['pregunta']}")
            print(f"Respuesta: {match['respuesta']}")
