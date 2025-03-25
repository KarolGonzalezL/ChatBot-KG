import json
import os
import pyttsx3
import subprocess

from langchain.memory import ConversationSummaryBufferMemory
from modules.database import get_vectorstore
from modules.logger import logger
import logging
from groq import Groq

client = Groq(api_key="")

#import logging

# Configuración de logger
logger = logging.getLogger(__name__)

# Función para guardar el historial de la conversación
def save_conversation(user_input, response):
    """
    Guarda la conversación en un archivo de historial.
    """
    try:
        with open("conversation_history.txt", "a") as file:
            file.write(f"Usuario: {user_input}\n")
            file.write(f"Chatbot: {response}\n\n")
    except Exception as e:
        logger.error(f"Error al guardar el historial de la conversación: {e}")

# Función para obtener el contexto relevante del vectorstore
def get_relevant_context(user_input, vectorstore):
    """
    Obtiene el contexto más relevante de la base de datos vectorial (ChromaDB) para la entrada del usuario.
    """
    try:
        # Realiza una búsqueda de similitud en el vectorstore
        results = vectorstore.similarity_search(user_input, k=5)  # 'k' indica cuántos documentos obtener

        # Combina los resultados del vectorstore en un solo contexto
        context = "\n".join([doc.page_content for doc in results])
        return context

    except Exception as e:
        logger.error(f"Error al obtener el contexto relevante: {e}")
        return "Lo siento, no pude obtener el contexto de la base de datos."

# Función para generar el prompt
def create_prompt(user_input, vectorstore_context):
    """
    Crea un prompt estructurado para el modelo de respuesta, combinando la entrada del usuario y el contexto relevante respondiendo de una manera cariñosa, amable y respetuosa.
    """
    prompt = f"""
    Basado en el contexto obtenido de la base de datos, responde a la siguiente consulta del usuario:
    
    Contexto relevante:
    {vectorstore_context}

    Pregunta del usuario:
    {user_input}

    Responde de manera amable, respetuosa, cariñosa , precisa y amorosa.
    """
    return prompt

def create_prompt_precios(user_input, vectorstore_context):
    prompt = f"""
    Basado en la información proporcionada sobre los efectos y precios de las extensiones de pestañas, responde a la siguiente consulta del usuario de manera clara y precisa:

    Usuario: {user_input}

    Aquí tienes los detalles de los diferentes efectos y precios:

    Montajes de Extensiones de Pestañas:
    1. **Clásicas Efecto Natural**: Un look suave, ideal para quienes prefieren un toque natural pero con volumen. ✨
    - Precio: $95,000 💖

    2. **Clásicas Efecto Pestañina**: Para un toque más coqueto, como si tuvieras la pestañina puesta todo el tiempo. 😍
    - Precio: $95,000 💕

    3. **Efecto Húmedo**: Un look brillante y fresco, como si tus pestañas estuvieran recién mojadas. 💦
    - Precio: $102,000 ✨

    4. **Efecto 2D**: Volumen adicional para un efecto más impactante y voluminoso, pero sin exagerar. 😘
    - Precio: $100,000 💕

    5. **Fibras Tecnológicas**: Para un volumen increíble con fibras innovadoras que dan un toque único. 🌟
    - Precio: $115,000 💖

    6. **Volumen Ruso**: ¡El máximo volumen y drama! Ideal si buscas una mirada súper impactante y volumétrica. 💣
    - Precio: $115,000 ✨

    7. **Híbridas**: Combinación perfecta entre volumen y naturalidad, ideal para una mirada equilibrada y hermosa. 😍
    - Precio: $110,000 💕

    Precios de Retoques (cada 15-20 días):
    1. **Retoques Pestañas Clásicas Efecto Natural** - Precio: $45,000 💖
    2. **Retoques Pestañas Clásicas Efecto Pestañina** - Precio: $45,000 💕
    3. **Retoques Pestañas Efecto Húmedo** - Precio: $50,000 🌸
    4. **Retoques Pestañas Efecto 2D** - Precio: $50,000 💖
    5. **Retoques Pestañas Fibras Tecnológicas** - Precio: $55,000 ✨
    6. **Retoques Pestañas Volumen Ruso** - Precio: $55,000 💕
    7. **Retoques Pestañas Híbridas** - Precio: $50,000 💖

    ¿Qué opción te gusta más, linda? 💖

    Contexto relevante:
    {vectorstore_context}
    """
    return prompt



def create_prompt_retiro(user_input, vectorstore_context):
    prompt = f"""
    Basado en la información proporcionada sobre el retiro de extensiones de pestañas, responde de manera amable, linda, cariñosa y clara a la siguiente consulta del usuario:

    Usuario: {user_input}

    Hermosa Aquí tienes los detalles sobre el costo del retiro de pestañas:

    - Si necesitas retirar extensiones de otro trabajo previo, el costo por el desmontaje es de **$15,000**. 
    - Recuerda que este servicio solo es necesario si ya tienes extensiones de otro profesional y deseas hacer un nuevo montaje con nosotros.

    Si tienes alguna pregunta adicional o necesitas más información, no dudes en preguntarnos. ¡Estamos aquí para ayudarte, bella! 💖

    Contexto relevante:
    {vectorstore_context}
    
    Responde de manera amigable y respetuosa, manteniendo un tono cariñoso hacia una mujer.
    """
    return prompt

def create_prompt_ubicacion(user_input, vectorstore_context):
    prompt = f"""
    Basado en la información proporcionada sobre nuestra ubicación, responde de manera clara amorosa, amigable y precisa a la siguiente consulta del usuario:

    Usuario: {user_input}

    Aquí tienes los detalles sobre nuestra ubicación:

    - Estamos ubicadas en **Cra 58BB #25-23 Bello Cabañas**. 🌸
    - Si tienes alguna dificultad para llegar o necesitas más detalles, no dudes en pedirnos más información.

    Si necesitas ayuda adicional o quieres saber cómo llegar, ¡aquí estamos para asistirte con mucho cariño! 💖

    Contexto relevante:
    {vectorstore_context}
    
    Responde de manera amigable y respetuosa, manteniendo un tono cariñoso .
    """
    return prompt

def create_prompt_cuidado_pestanas(user_input, vectorstore_context):
    prompt = f"""
    Basado en la información proporcionada sobre el cuidado de las extensiones de pestañas, responde de manera clara, cariñosa, respetuosa, amigable y precisa a la siguiente consulta del usuario:

    Usuario: {user_input}

    Aquí tienes algunos consejos para cuidar tus extensiones de pestañas y mantenerlas hermosas por más tiempo:

    - Evita productos grasosos al desmaquillarte o limpiar tu rostro.
    - No mojes las pestañas con agua durante las primeras 24 horas después del procedimiento.
    - Evita saunas, turcos y limpiezas faciales con vapor.
    - Al lavarte el rostro, no uses agua caliente y evita el chorro fuerte de la ducha sobre tus ojos.
    - No duermas boca abajo ni frotes tus ojos bruscamente.
    - No apliques pestañina, ya que acelera la caída de las extensiones.
    - No uses encrespadores, cucharas ni latas, ya que dañan las extensiones y las naturales.
    - No tintures ni maquilles las extensiones.
    - Evita arrancar las extensiones para no dañar tus pestañas naturales.

    Recuerda que si tienes 100 pestañas naturales fuertes, puedes colocar hasta 100 extensiones. ¡Cuida tus pestañas y manténlas perfectas! 💖

    Contexto relevante:
    {vectorstore_context}
    
    Responde de manera amigable y cariñosa, manteniendo un tono claro y preciso.
    """
    return prompt

def create_prompt_disponibilidad_agenda(user_input, vectorstore_context):
    
    prompt = f"""
    basado en la informacion sobre la disponibilidad de citas, responde de manera inteligente y creativa para dar varias opciones y despues pasar a otro prompt de agendamiento de citas
    Usuario: {user_input}
    
    💖 Qué lindo que quieras agendar tu cita con nosotros este fin de semana. Estoy aquí para ayudarte. 😊

    Para el fin de semana, tenemos varias opciones disponibles:

    - El **sábado por la mañana** a las 10:00 AM.
    - El **sábado por la tarde** a las 3:00 PM.
    - O el **domingo por la mañana** a las 11:00 AM.
    - También tenemos una opción en la **tarde del domingo** a las 4:00 PM.

    Por favor, dime cuál te viene mejor y con gusto agendamos tu cita. Si ninguno de estos horarios te funciona, no dudes en decirme y buscaremos otra opción que se ajuste a tu agenda. 😘💫

    ¡Espero tu respuesta con mucho cariño!
    Contexto relevante:
    {vectorstore_context}
    """
    return prompt

def create_prompt_agendar_cita(user_input, vectorstore_context):
    prompt = f"""
    Basado en la información proporcionada sobre la separación o agendamiento de citas, responde de manera amable y precisa a la siguiente consulta del usuario:

    Usuario: {user_input}

    Si el comprobante ha sido verificado correctamente, sigue este proceso para agendar la cita:
    
    1. Para confirmar tu cita, es necesario realizar un abono de $30,000 a la cuenta 📲 **91228348649**.
    2. Una vez realizado el pago, por favor sube el comprobante de la transferencia aquí, y con mucho gusto agendaremos tu cita. 💖

    Si el comprobante no ha sido verificado, por favor revisa los datos y vuelve a intentarlo.

    Contexto relevante:
    {vectorstore_context}

    Responde de manera amigable y respetuosa, manteniendo un tono cariñoso.
    """
    return prompt

import pyttsx3

# Inicialización del motor de voz
engine = pyttsx3.init()

# Función para responder en voz
def respond_with_voice(response):
    try:
        engine.say(response)
        engine.runAndWait()  # Espera a que termine de hablar antes de continuar
    except Exception as e:
        logger.error(f"Error al generar la respuesta de voz: {e}")

# Función para manejar la solicitud de Instagram o fotos

def create_prompt_instagram(user_input, vectorstore_context):
    """
    Este prompt responde a las consultas sobre Instagram enviando el enlace de la cuenta de Instagram.
    Además, el chatbot responderá en voz.
    """

    # Si el usuario menciona "Instagram" o algo relacionado, se responde con el enlace
    prompt = f"""
    Usuario: {user_input}

    Claro hermosa, aquí tienes el enlace de nuestro Instagram para que nos sigas y veas todos nuestros trabajos:

    Instagram: https://www.instagram.com/kgpestanas/ 💖

    ¡Espero que te encanten nuestras fotos!
    Contexto relevante:
    {vectorstore_context}
    responde de manera cariñosa
    """

    return prompt

def select_prompt(user_input, vectorstore_context):
    """
    Selecciona el prompt adecuado según la consulta del usuario.
    """
    if "cita" in user_input.lower():
        return create_prompt_agendar_cita(user_input, vectorstore_context)
    elif "retiro" in user_input.lower():
        return create_prompt_retiro(user_input, vectorstore_context)
    elif "ubicadas" in user_input.lower():
        return create_prompt_ubicacion(user_input, vectorstore_context)
    elif "cuidado" in user_input.lower():
        return create_prompt_cuidado_pestanas(user_input, vectorstore_context)
    elif "precios" in user_input.lower():
        return create_prompt_precios(user_input, vectorstore_context)
    elif "disponibilidad" in user_input.lower() or "tienes cita" in user_input.lower():
        return create_prompt_disponibilidad_agenda(user_input, vectorstore_context)
    elif "link" in user_input.lower() or "fotos" in user_input.lower():
        return create_prompt_instagram(user_input, vectorstore_context)
    else:
        return create_prompt(user_input, vectorstore_context)

# chatbot.py


def check_for_goodbye(user_input):
    """
    Función para detectar despedidas y finalizar la conversación.
    """
    # Palabras clave que el usuario podría usar para despedirse
    goodbye_keywords = ["adiós", "hasta luego", "nos vemos", "chao", "gracias", "hasta pronto", "me voy"]

    # Convertir el texto del usuario a minúsculas para hacer una comparación insensible a mayúsculas
    user_input = user_input.lower()

    # Verificar si el usuario está diciendo alguna de las despedidas
    if any(goodbye in user_input for goodbye in goodbye_keywords):
        return True
    return False

engine = pyttsx3.init()


def chat_with_groq(user_input, vectorstore, image_verified):
    try:


        # Comprobar si el usuario está despidiéndose
        if check_for_goodbye(user_input):
            response = "¡Gracias a ti hermosa! ¡Hasta la próxima! Que tengas un excelente día."
            save_conversation(user_input, response)

            return response# Responder en voz

            # Responder en texto también
        
        # Si no es despedida, procesar el contexto y generar la respuesta en texto
        context = get_relevant_context(user_input, vectorstore)
        if image_verified:
            context += "\n\nImagen verificada correctamente."
        
        prompt = select_prompt(user_input, context)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            model="llama3-70b-8192",
        )
        response = chat_completion.choices[0].message.content
        save_conversation(user_input, response)
        
        # Responder solo en texto para otras preguntas
        print(response)  # Asegurarse de que la respuesta también aparezca en pantalla
        return response

    except Exception as e:
        logger.error(f"Error al generar la respuesta: {e}")
        return "Lo siento, hubo un error al procesar tu solicitud."
