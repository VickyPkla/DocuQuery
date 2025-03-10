import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set Streamlit page config
st.set_page_config(page_title="DocuQuery AI", page_icon="🤖", layout="wide")

# Title with style
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>DocuQuery AI: Q&A with Groq-Powered LLMs</h1>
    """, unsafe_allow_html=True)
    #<p style='text-align: center; color: gray;'>Ask questions based on uploaded documents</p>

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Provide the most accurate response.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to create vector embeddings
def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        with st.spinner("Processing and embedding documents..."):
            st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            documents = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join("temp_uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("✅ Vector Store DB is ready!")

# Sidebar for document embedding
with st.sidebar:
    st.subheader("📄 Upload Documents for Embeddings")
    uploaded_files = st.file_uploader("", accept_multiple_files=True, type=["pdf"])
    if st.button("Create Embeddings") and uploaded_files:
        os.makedirs("temp_uploads", exist_ok=True)
        vector_embedding(uploaded_files)
    
# User input section
st.subheader("💡 Ask a Question About Your Documents")
input_text = st.text_input("")

# Process query if input exists
if input_text:
    if "vectors" not in st.session_state:
        st.error("❌ No document embeddings found! Please upload and process documents first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)
        
        with st.spinner("🔎 Searching for the most relevant answer..."):
            start = time.process_time()
            response = retriever_chain.invoke({"input": input_text})
            elapsed_time = time.process_time() - start
            
        # Display answer
        st.success("✅ Answer Found!")
        st.markdown(f"**Response:** {response['answer']}")
        st.caption(f"⏳ Processed in {elapsed_time:.2f} seconds")
        
        # Document similarity search display
        with st.expander("📚 Relevant Document Chunks"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")
