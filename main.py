from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path
from pdfplumber import open as pdf_open
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI(title="RAG Ingestion API")

# 🔑 Configuración: Usa variables de entorno (NO pongas claves aquí)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-uowwgk0owg4cwccs8skow4cg:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

COLLECTION_NAME = "documents"

# Cargar modelo de embeddings
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae texto de un PDF (solo funciona con PDFs de texto, no escaneados)"""
    text = ""
    with pdf_open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def clean_and_split(text: str) -> list:
    """Divide el texto en fragmentos de tamaño razonable"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_text(text)

def ensure_collection_exists():
    """Crea la colección en Qdrant si no existe"""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    # Validar que sea un archivo PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    # Guardar el archivo temporalmente en el contenedor
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Extraer texto del PDF
        text = extract_text_from_pdf(tmp_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF has no extractable text (might be scanned)")

        # Dividir en fragmentos
        chunks = clean_and_split(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks created from PDF")

        # Generar embeddings para cada fragmento
        embeddings = embedder.encode(chunks)

        # Asegurar que la colección exista en Qdrant
        ensure_collection_exists()

        # Preparar los puntos para Qdrant
        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=abs(hash(chunk)) % (2**32),  # ID numérico estable
                    vector=emb.tolist(),
                    payload={
                        "text": chunk,
                        "source": file.filename,
                        "chunk_index": i
                    }
                )
            )

        # Guardar en Qdrant
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return JSONResponse({
            "status": "success",
            "chunks_inserted": len(points),
            "source": file.filename
        })

    finally:
        # Eliminar el archivo temporal
        os.unlink(tmp_path)