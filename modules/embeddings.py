from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from modules.database import get_vectorstore
from modules.logger import logger

def load_and_store_documents(file_content, file_name):
    """
    Carga documentos subidos a través de Streamlit, genera embeddings y los almacena en ChromaDB.
    """
    try:
        ruta_guardado = f"data/documentos/{file_name}"
        
        # Guardar el archivo correctamente
        with open(ruta_guardado, "wb") as f:
            f.write(file_content.getvalue())  # Usamos getvalue() para obtener el contenido como bytes

        logger.info(f"Archivo guardado en: {ruta_guardado}")

        # Intentar cargar el archivo con TextLoader
        try:
            loader = TextLoader(ruta_guardado)
            documents = loader.load()
            logger.info(f"Documento {file_name} cargado correctamente.")
        except Exception as e:
            logger.error(f"Error al cargar el archivo {file_name} con TextLoader: {e}")
            return

        # Intentar dividir los documentos en chunks
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            logger.info(f"Documento {file_name} dividido correctamente en chunks.")
        except Exception as e:
            logger.error(f"Error al dividir el documento {file_name} en chunks: {e}")
            return

        # Intentar agregar los documentos al vectorstore
        try:
            vectorstore = get_vectorstore()
            if vectorstore:
                vectorstore.add_documents(docs)
                logger.info(f"Documentos del archivo {file_name} indexados correctamente en ChromaDB.")
                logger.info(f"Contenido actual del vectorstore: {vectorstore}")
            else:
                logger.warning("No se pudo indexar el documento debido a un problema con la base de datos vectorial.")
        except Exception as e:
            logger.error(f"Error al agregar documentos al vectorstore: {e}")
            return

    except Exception as e:
        logger.error(f"Error al cargar y almacenar documentos: {e}")
