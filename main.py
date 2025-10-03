FROM python:3.10-slim

# Evitar prompts y mejorar rendimiento de pip
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Paquetes del sistema (lo mínimo). Si algún wheel necesita compilar, deja build-essential.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias primero para cachear
COPY requirements.txt .

# (Opcional) fija torch CPU para evitar sorpresas
# Si prefieres la última, deja solo "pip install -r requirements.txt"
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

# Descargar el modelo de embeddings durante la build (cacheable)
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-en-v1.5')
PY

# Copiar el código
COPY . .

EXPOSE 8765

# Uvicorn en modo producción sencillo; si necesitas más concurrencia, añade --workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]
