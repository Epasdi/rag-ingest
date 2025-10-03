FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar el modelo de embeddings DURANTE la construcción
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copiar el código
COPY . .

# Exponer puerto
EXPOSE 8765

# Ejecutar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]