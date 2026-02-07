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

# 2. Configuración de la página
st.set_page_config(page_title="Mi Portafolio AI - RAG", layout="wide")
st.title("📄 Chat con tus Documentos (RAG)")
load_dotenv("../.env")

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

# 6. Área Principal: Chat
st.divider() # Una línea visual para separar
pregunta_usuario = st.chat_input("Pregúntale algo a tu documento...")

if pregunta_usuario:
    # Verificamos si ya tenemos el archivo procesado en la base de datos
    if 'retriever' in locals():
        with st.spinner("Buscando en el documento..."):
            # 7. Configurar el LLM (Gemini)
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
            
            # 8. Definir el "Plano de construcción" del RAG (Prompt)
            template = """Responde la pregunta basándote solo en el contexto proporcionado:
            <contexto>
            {context}
            </contexto>
            Pregunta: {question}"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # 9. La Cadena RAG (Usando la sintaxis moderna que vimos en la doc)
            # Esta cadena: toma la pregunta -> busca contexto -> se lo pasa al LLM -> devuelve texto
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # 10. Ejecutar y mostrar respuesta
            respuesta = chain.invoke(pregunta_usuario)
            st.markdown(f"### Respuesta:\n{respuesta}")
    else:
        st.warning("⚠️ Primero sube un archivo en la barra lateral para poder chatear.")