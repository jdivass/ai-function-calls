# Hoja de Trabajo 4 - Function calls

## Infraestructura PostgreSQL + pgvector

La solución utiliza PostgreSQL 16 con la extensión `pgvector`, ejecutado en
Docker Compose. Los embeddings generados con `all-MiniLM-L6-v2` deben tener
384 dimensiones y se almacenan en la tabla `faq_embeddings`.

### Requisitos

- Docker Engine y Docker Compose v2 (`docker compose`)
- Python 3.10 o superior

### Inicialización

Desde este directorio (`ai-function-calls`):

```bash
cp .env.example .env
# Edite .env y establezca una contraseña segura en POSTGRES_PASSWORD.
docker compose up -d
docker compose ps
```

Instale las dependencias del cargador:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El primer arranque crea automáticamente la extensión `vector`, la tabla y sus
índices mediante `docker/init.sql`. El volumen `parachute-pgdata` conserva los
datos aunque el contenedor se detenga o se recree.

Para revisar el estado o los logs:

```bash
docker compose logs -f postgres
docker compose exec postgres pg_isready -U parachute -d parachute_faqs
```

La conexión desde el equipo host usa los valores de `.env`:

```text
host=localhost port=5432 dbname=parachute_faqs user=parachute password=<POSTGRES_PASSWORD>
```

### Cargar el corpus

Con PostgreSQL iniciado y el entorno virtual activo:

```bash
python src/load_corpus.py
```

El script parsea las 120 FAQs del archivo
`data/Corpus_FAQs_Parachute_SA_2026.txt`,
genera embeddings de 384 dimensiones con `all-MiniLM-L6-v2` y ejecuta un
`upsert` usando el `id` de cada FAQ. Por eso se puede ejecutar nuevamente sin
crear registros repetidos; si cambia una FAQ, su fila y embedding se actualizan.

### Estructura

```text
ai-function-calls/
├── data/                 # Corpus de conocimiento
├── docker/               # Script de inicialización de PostgreSQL
├── src/                  # Código ejecutable
│   └── load_corpus.py
├── docker-compose.yml
├── requirements.txt
└── .env
```

### Reinicializar la base de datos

`init.sql` solo se ejecuta cuando PostgreSQL inicializa un volumen vacío. Para
volver a crear el esquema desde cero, detenga el servicio y elimine únicamente
el volumen de este proyecto:

```bash
docker compose down
docker compose down -v
docker compose up -d
```

Esto elimina los datos almacenados en el volumen definido por este proyecto.
