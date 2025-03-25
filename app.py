import streamlit as st
import os
import time
import cv2
import pytesseract as tess
import re
from PIL import Image
from modules.chatbot import chat_with_groq  
from modules.embeddings import load_and_store_documents
from modules.database import get_vectorstore
from modules.logger import logger
import numpy as np
import speech_recognition as sr
import pyttsx3


# Configurar Tesseract (Asegúrate de que esté instalado en tu sistema)
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Configuración de la página de Streamlit
import streamlit as st

# Cambiar colores utilizando CSS
st.set_page_config(page_title="CHAT KG ESTUDIO", layout="wide")
st.title("Bienvenida a KG ESTUDIO, ¿En qué te podemos ayudar? 🌸")



# Función para procesar archivos de texto
def process_text_file(uploaded_file, file_name):
    file_path = f"data/documentos/{file_name}"
    if not os.path.exists(file_path) or os.path.getmtime(file_path) < time.time():
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        load_and_store_documents(uploaded_file, file_name)  # Reindexar documento
        logger.info(f"Archivo {file_name} cargado e indexado correctamente.")
        st.sidebar.success(f"Archivo {file_name} cargado correctamente y agregado al contexto.")
    else:
        logger.info(f"El archivo {file_name} ya ha sido indexado previamente.")
        st.sidebar.info(f"El archivo {file_name} ya ha sido indexado previamente.")

st.sidebar.header("Carga de Documentos")
uploaded_document = st.sidebar.file_uploader("Sube un documento (TXT, CSV, JSON)", type=["txt", "csv", "json"])

# Procesar archivos subidos
if uploaded_document is not None:
    file_name = uploaded_document.name
    if file_name.lower().endswith(('.txt', '.csv', '.json')):
        process_text_file(uploaded_document, file_name)

# Función para escuchar y transcribir lo que se dice
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def talk():
    mic = sr.Microphone()
    with mic as source:
        audio = recognizer.listen(source)
    text = recognizer.recognize_google(audio, language='ES')    
    
    print(f"Has dicho: {text}")
    return text.lower()
def respond_with_voice(response):
    try:
        engine.say(response)
        engine.runAndWait()
    except Exception as e:
        print(f"Error al generar la respuesta de voz: {e}")

# Función que detecta si la entrada es una despedida
def check_for_goodbye(text):
    goodbye_keywords = ['adiós', 'hasta luego','muchas gracias','gracias' 'nos vemos', 'chao', 'bye']
    return any(word in text for word in goodbye_keywords)

def process_image_and_verify(uploaded_file):
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Convertir la imagen a formato OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Extraer texto con OCR
    extracted_text = tess.image_to_string(image_cv)

    # Limpiar el texto extraído
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[áàäâ]', 'a', text)
        text = re.sub(r'[éèëê]', 'e', text)
        text = re.sub(r'[íìïî]', 'i', text)
        text = re.sub(r'[óòöô]', 'o', text)
        text = re.sub(r'[úùüû]', 'u', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Eliminar caracteres especiales
        return text

    cleaned_text = clean_text(extracted_text)

    # Palabras clave para verificar la transferencia
    keywords = ['transferencia exitosa', 'comprobante no', 'valor de la transferencia', '30000', 'karol dayanna gonzalez lopez']

    # Verificar si las palabras clave están presentes
    if all(keyword in cleaned_text for keyword in keywords):
        
        return True  # Indicamos que la validación fue exitosa
    else:
        
        return False  # Indicamos que la validación falló

# Sidebar para subir imágenes
st.sidebar.header("Carga de Imágenes")
uploaded_image = st.sidebar.file_uploader("Sube la imagen de la transferencia", type=["jpg", "png"])
if "messages" not in st.session_state:
    st.session_state.messages = []

# Guardar historial antes de cargar imagen
def update_chat_with_image(image_path):
    st.session_state.messages.append({"role": "assistant", "content": f"Imagen cargada: {image_path}"})


# Mostrar historial de conversación
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Llamada a la función de verificación si se sube una imagen

if uploaded_image:
    validation_result = process_image_and_verify(uploaded_image)

    # Si la validación es exitosa, mostramos el mensaje de confirmación y agendamiento de cita
    if validation_result:
        # Mensaje de éxito cuando la validación es correcta
        st.session_state.messages.append({"role": "assistant", "content": "Listo hermosa, tu cita ha sido agendada. Un día antes te escribimos para recordarte la cita."})
        
        # Mostrar el mensaje en el chat
        with st.chat_message("assistant"):
            st.markdown("Listo hermosa, tu cita ha sido agendada. Un día antes te escribimos para recordarte la cita.")
    else:
        # Si la validación falla, mostramos un mensaje de error
        st.session_state.messages.append({"role": "assistant", "content": "No se pudo validar correctamente el comprobante. Intenta nuevamente."})
        
        # Mostrar el mensaje de error en el chat
        with st.chat_message("assistant"):
            st.markdown("No se pudo validar correctamente el comprobante. Intenta nuevamente.")





# Entrada del usuario
# Entrada del usuario
user_input = st.chat_input("Escribe tu mensaje...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Verificar si la base de datos vectorial está cargada en la sesión
    if "vectorstore" in st.session_state and st.session_state.vectorstore:
        # Si el vectorstore está cargado, procesar la respuesta
        response = chat_with_groq(user_input, st.session_state.vectorstore,image_verified=process_image_and_verify)
    else:
        # Si el vectorstore no está disponible, mostrar error y recargar
        response = "Error: No se pudo recuperar la base de datos vectorial."

        st.session_state.vectorstore = get_vectorstore()  # Recargar el vectorstore

        # Verificar si el vectorstore se cargó correctamente
        if st.session_state.vectorstore:
            response = chat_with_groq(user_input, st.session_state.vectorstore,image_verified=process_image_and_verify)
        else:
            response = "Error crítico: No se pudo recuperar la base de datos vectorial."
            st.error("Error crítico: No se pudo recuperar la base de datos vectorial. Verifique la conexión con ChromaDB.")

    # Mostrar la respuesta generada
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)



if st.button("Hablar respuesta"):
    st.session_state.messages.append({"role": "assistant", "content": "Ahora escuchando... por favor habla."})
    with st.chat_message("assistant"):
        st.markdown("¡Escuchando...")

    # Llamar a la función de voz para transcribir lo que dice el usuario
    try:
        transcribed_input = talk()  # Obtiene el texto transcrito de la voz
        if transcribed_input:  # Si hay texto transcrito
            st.session_state.messages.append({"role": "user", "content": transcribed_input})  # Agregar la transcripción al chat
            with st.chat_message("user"):
                st.markdown(transcribed_input)

            # Procesar la nueva entrada del usuario (transcrita)
            if check_for_goodbye(transcribed_input):  # Si es una despedida
                response = "¡Gracias a ti hermosa! ¡Hasta la próxima! Que tengas un excelente día."
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Responder en voz a la despedida
                respond_with_voice(response)

            else:
                # Si no es despedida, procesamos normalmente
                if "vectorstore" in st.session_state and st.session_state.vectorstore:
                    response = chat_with_groq(transcribed_input, st.session_state.vectorstore, image_verified=process_image_and_verify)
                else:
                    response = "Error: No se pudo recuperar la base de datos vectorial."
                    st.session_state.vectorstore = get_vectorstore()  # Recargar el vectorstore
                    if st.session_state.vectorstore:
                        response = chat_with_groq(transcribed_input, st.session_state.vectorstore, image_verified=process_image_and_verify)
                    else:
                        response = "Error crítico: No se pudo recuperar la base de datos vectorial."
                        st.error("Error crítico: No se pudo recuperar la base de datos vectorial. Verifique la conexión con ChromaDB.")

                # Mostrar la respuesta generada para la nueva entrada
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

                # Responder en voz con la respuesta generada por el chatbot


    except Exception as e:
        st.error(f"Hubo un error al transcribir tu voz: {e}")