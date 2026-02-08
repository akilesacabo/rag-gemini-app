import streamlit as st
import os
from typing import List, Optional
from dotenv import load_dotenv

# Importaciones de LangChain y Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURACIÓN Y CONSTANTES ---
# Centralizamos las configuraciones para no tener "números mágicos" dispersos en el código.
CONFIG = {
    "page_title": "Portafolio AI - RAG Pro",
    "page_icon": "🧠",
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "embedding_model": "gemini-embedding-001",
    "llm_model": "gemini-3-flash-preview",
    "k_retrieval": 5  # Número de fragmentos a recuperar
}

# Cargar variables de entorno una sola vez al inicio
load_dotenv()


# --- 2. Backend ---

def validate_api_key(key_name ="GOOGLE_API_KEY" ):
    """Verifica la existencia de la API Key en las variables de entorno"""
    if not os.getenv(key_name):
        st.error("❌ ERROR CRÍTICO: No se encontró GOOGLE_API_KEY en el archivo .env")
        st.stop()


@st.cache_resource(show_spinner=True)
def process_document_to_vectorstore(file_content: str):
    """Procesa el texto y crea la base de datos vectorial."""
    
    try: 
        #1 Spliting 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CONFIG["chunk_size"],
            chunk_overlap = CONFIG["chunk_overlap"]
        )
        fragments  = text_splitter.split_text(file_content)

        #2 Embedding 
        embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

        #3 Vector Store
        vector_store = Chroma.from_texts(
            texts=fragments,
            embedding=embeddings_model
            )
        return vector_store

    except Exception as e :
        st.error(f"Error procesando el documento: {e}")
        return None
    
def get_rag_chain(retriever,historial_str) -> Runnable:
    """Construye el pipeline de procesamiento LCEL."""
    
    llm = ChatGoogleGenerativeAI(model=CONFIG["llm_model"])
    template = """Eres un asistente experto y preciso.
    Usa la siguiente información de contexto para responder a la pregunta del usuario.
    Si la respuesta no está en el contexto, di que no lo sabes.

    CONTEXTO:
    {context}

    HISTORIAL:
    {chat_history}

    PREGUNTA: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    # Construcción de la cadena LCEL
    chain = (
            {
                # CAMBIO AQUÍ: Le decimos que para el contexto solo use la 'question' del diccionario
                "context": retriever,
                
                # Aquí también extraemos solo la pregunta para el prompt
                "question": RunnablePassthrough(),
                
                # Y aquí el historial
                "chat_history": lambda x: historial_str
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    
    return chain

# --- 3. FRONTEND ---

def initialize_session_state():
    """Inicializa las variables de sesión si no existen."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

def render_sidebar():
    """Gestiona la barra lateral y la carga de archivos."""
    with st.sidebar:
        st.header("📂 Configuración")
        uploaded_file = st.file_uploader("Sube tu documento (.txt)", type=["txt"])
        
        if uploaded_file:
            # Leemos el archivo
            string_data = uploaded_file.read().decode("utf-8")
            
            # Llamamos a la función con caché
            # Si el archivo es el mismo, Streamlit devuelve el resultado guardado instantáneamente.
            vectorstore = process_document_to_vectorstore(string_data)
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.success(f"✅ Documento procesado ({len(string_data)} caracteres)")
        
        if st.button("🧹 Limpiar Chat"):
            st.session_state.messages = []
            st.rerun()

def render_chat_interface():
    """Dibuja el historial y gestiona el input del usuario."""
    st.title(CONFIG["page_title"])

    # 1. Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Input de usuario
    if prompt := st.chat_input("Escribe tu pregunta sobre el documento..."):
        # Guardar y mostrar pregunta
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Generar respuesta
        if st.session_state.vectorstore is not None:
            process_answer(prompt)
        else:
            st.warning("⚠️ Por favor, sube un documento primero para poder analizarlo.")

def process_answer(question: str):
    """Lógica para generar y mostrar la respuesta del asistente."""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Recuperar historial formateado
        # Tomamos los últimos 5 pares para no saturar el contexto
        history_str = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-10:]]
        )
        
        # Configurar retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": CONFIG["k_retrieval"]}
        )
        
        # Obtener la cadena
        rag_chain = get_rag_chain(retriever,history_str)
        
        try:
            # Ejecutar la cadena pasando el historial correctamente
            response = rag_chain.invoke(question)
            
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Error generando respuesta: {e}")

# --- 4. MAIN ENTRY POINT ---
def main():
    st.set_page_config(
        page_title=CONFIG["page_title"], 
        page_icon=CONFIG["page_icon"], 
        layout="wide"
    )
    validate_api_key()
    initialize_session_state()
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()