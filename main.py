# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import tempfile
import os
import hashlib
from typing import List
from pdfplumber import open as pdf_open
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI(title="RAG Ingestion API")

# =========================
# Configuración por entorno
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # opcional si Qdrant sin auth
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")

# =====================================
# Modelo de embeddings y cliente Qdrant
# =====================================
# Carga única al arrancar el proceso
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
# bge-small-en-v1.5 -> 384 dimensiones
EMBED_DIM = 384

# Usa timeouts prudentes para evitar cuelgues
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)


# ==============
# Utilidades
# ==============
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae texto de un PDF (no OCR)."""
    text = ""
    with pdf_open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text


def clean_and_split(text: str) -> List[str]:
    """Divide el texto en fragmentos."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_text(text)


def ensure_collection_exists():
    """Crea la colección si no existe. Tolerante a carreras."""
    try:
        if not qdrant.collection_exists(COLLECTION_NAME):
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
    except Exception:
        # Si falló por condición de carrera, vuelve a comprobar
        if not qdrant.collection_exists(COLLECTION_NAME):
            raise


def stable_point_id(text: str) -> int:
    """ID entero estable a partir de SHA1 del chunk."""
    h = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")  # 64 bits


# ==========
# Endpoints
# ==========
@app.get("/health")
def health():
    """Comprueba conectividad con Qdrant."""
    try:
        _ = qdrant.get_collections()
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(
            {"status": "qdrant_unreachable", "error": str(e)}, status_code=503
        )


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingerir un PDF y almacenarlo en Qdrant como chunks embebidos."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Guarda temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Extrae texto
        text = extract_text_from_pdf(tmp_path)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="PDF has no extractable text (might be scanned)",
            )

        # Split
        chunks = clean_and_split(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks created from PDF")

        # Embeddings
        embeddings = embedder.encode(chunks).tolist()

        # Colección
        ensure_collection_exists()

        # Upsert
        points = [
            PointStruct(
                id=stable_point_id(chunk),
                vector=emb,
                payload={"text": chunk, "source": file.filename, "chunk_index": i},
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return {
            "status": "success",
            "chunks_inserted": len(points),
            "source": file.filename,
            "collection": COLLECTION_NAME,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


@app.get("/search")
def search(
    q: str = Query(..., description="Consulta en lenguaje natural"),
    top_k: int = Query(5, ge=1, le=50, description="Resultados a devolver"),
):
    """Busca en Qdrant usando embeddings del query."""
    # Vectoriza query
    vec = embedder.encode([q])[0].tolist()

    # Asegura que la colección exista (por si se llama antes de /ingest)
    ensure_collection_exists()

    # Búsqueda
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=top_k,
    )

    # Normaliza a JSON simple
    return [
        {
            "score": float(res.score),
            "text": (res.payload or {}).get("text"),
            "source": (res.payload or {}).get("source"),
            "chunk_index": (res.payload or {}).get("chunk_index"),
        }
        for res in results
    ]
