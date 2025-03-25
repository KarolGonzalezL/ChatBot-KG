
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging
from modules.logger import logger


# Variable global para almacenar el modelo
embeddings = None

def get_vectorstore():
    global embeddings
    if embeddings is None:
        try:
            logger.info("Cargando modelo de embeddings...")  # Solo carga el modelo la primera vez
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Error al cargar el modelo de embeddings: {e}")
            return None
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)