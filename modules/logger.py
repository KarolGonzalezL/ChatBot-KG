import logging

# Configuración del sistema de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Mostrar logs en consola
        logging.FileHandler("app.log")  # Guardar logs en un archivo
    ]
)

logger = logging.getLogger(__name__)