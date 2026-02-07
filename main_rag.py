import streamlit as st
import os
from dotenv import load_dotenv

# 1. IMPORTACIONES BASADAS EN LA DOCUMENTACIÓN OFICIAL (RAG + CHROMA)
# No usamos 'langchain.chains' porque, como bien dices, está deprecado/movido
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 2. Configuración de la página
st.set_page_config(page_title="Mi Portafolio AI - RAG", layout="wide")
st.title("📄 Chat con tus Documentos (RAG)")
load_dotenv()

# 3. Sidebar para subir archivos
with st.sidebar:
    st.header("Configuración")
    archivo_subido = st.file_uploader("Sube un archivo de texto (.txt)", type=("txt"))
    
    if archivo_subido:
        st.success("Archivo subido con éxito")

        # Mostrar un mensaje de error si el usuario sube un archivo incorrecto

        # 1. Leer el contenido del archivo
        # El archivo subido viene en formato binario, por eso usamos .read().decode()
        string_data = archivo_subido.read().decode("utf-8")

        # 2. Configurar el "Splitter" (El que corta el texto)
        # Usamos el RecursiveCharacterTextSplitter por recomendación oficial:
        # Intenta mantener párrafos y oraciones juntos para no perder el sentido.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,    # Tamaño de cada trozo (caracteres)
            chunk_overlap=200   # Cuánto se repite entre un trozo y otro
        )

        # 3. Crear los fragmentos (chunks)
        # Convertimos el texto largo en una lista de pedacitos
        fragmentos = text_splitter.split_text(string_data)

        # Mostrar una pequeña estadística en la interfaz
        st.info(f"El documento ha sido dividido en {len(fragmentos)} fragmentos.")

        # 4. Crear los Embeddings y la Base de Datos Vectorial
        # Usamos el modelo de Google para generar los vectores numéricos
        embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        
        # Creamos la base de datos Chroma en memoria a partir de nuestros fragmentos
        vectorstore = Chroma.from_texts(
            texts=fragmentos, 
            embedding=embeddings_model
        )
        
        st.success("✨ Base de datos vectorial creada correctamente")

        # 5. Configurar el "Retriever" (Buscador)
        # Esta pieza es la que buscará los fragmentos más parecidos a la pregunta del usuario
        retriever = vectorstore.as_retriever()

# Definición del modelo usando la versión 3.0 Preview
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# --- 1. GESTIÓN DE MEMORIA (Session State) ---
# Inicializamos el "baúl" de mensajes si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Dibujamos los mensajes previos en cada "rerun" de la app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 2. ENTRADA DE USUARIO ---
if prompt_usuario := st.chat_input("¿En qué puedo ayudarte con el documento?"):
    # Añadimos y mostramos la pregunta del usuario
    st.chat_message("user").markdown(prompt_usuario)
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})

    # --- 3. PROCESAMIENTO RAG CON CONTEXTO ---
    if 'retriever' in locals() or 'retriever' in globals():
        with st.chat_message("assistant"):
            # Serializamos el historial a String (últimos 6 mensajes para no saturar)
            historial_str = "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-7:-1]]
            )

            # Definimos el Template con el placeholder de historial
            template = """Eres un asistente experto. Responde de forma concisa usando el contexto e historial.
            
            CONTEXTO RECUPERADO:
            {context}
            
            HISTORIAL DE CHARLA:
            {chat_history}
            
            PREGUNTA: {question}
            """
            
            prompt_template = ChatPromptTemplate.from_template(template)
            
            # La "Tubería" (Chain) con el truco de la función Lambda
            chain = (
                {
                    "context": retriever, 
                    "question": RunnablePassthrough(),
                    "chat_history": lambda x: historial_str
                }
                | prompt_template
                | llm 
                | StrOutputParser()
            )

            # Generamos respuesta
            respuesta = chain.invoke(prompt_usuario)
            st.markdown(respuesta)
            
            # Guardamos la respuesta en la memoria
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
    else:
        st.warning("Por favor, sube un documento primero para activar el Retriever.")