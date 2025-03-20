import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
import os
import uuid
import time
import threading
import hashlib
import pandas as pd
import logging
import tempfile

# ======= 📝 Konfigurasi Logging =======
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======= 🚀 Konfigurasi Streamlit =======
st.set_page_config(
    page_title="AI Business Consultant Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======= 🔒 Fungsi Keamanan =======
def sanitize_filename(filename):
    """Sanitasi nama file untuk keamanan"""
    return hashlib.md5(filename.encode()).hexdigest()

def get_secure_temp_file(uploaded_file):
    """Membuat file sementara dengan nama yang aman"""
    temp_dir = tempfile.gettempdir()
    secure_filename = sanitize_filename(uploaded_file.name)
    extension = os.path.splitext(uploaded_file.name)[1]
    temp_file_path = os.path.join(temp_dir, f"{secure_filename}{extension}")

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Menambahkan file ke cleanup list
    if "temp_files" not in st.session_state:
        st.session_state.temp_files = []
    st.session_state.temp_files.append(temp_file_path)

    return temp_file_path

# ======= 🧹 Fungsi Pembersihan =======
def cleanup_temp_files():
    """Membersihkan file sementara saat sesi berakhir"""
    if "temp_files" in st.session_state:
        for file_path in st.session_state.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"File berhasil dihapus: {file_path}")
            except Exception as e:
                logger.error(f"Gagal menghapus file: {file_path} - {str(e)}")
        st.session_state.temp_files = []

# ======= 💼 Fungsi Inisialisasi =======
def initialize_app():
    """Inisialisasi aplikasi dan session state"""
    # Memastikan semua state tersedia
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "file_info" not in st.session_state:
        st.session_state.file_info = {}
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "memory" not in st.session_state:
        st.session_state.memory = None

# ======= 🔍 Load API Key =======
def load_api_key():
    """Load API key dari secrets atau input user"""
    # Prioritaskan dari secrets
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]

    # Jika tidak ada di secrets, minta dari user
    api_key = st.sidebar.text_input("🔑 OpenAI API Key:", type="password")
    if not api_key:
        st.sidebar.warning("⚠️ Silakan masukkan API Key OpenAI untuk memulai.")
    return api_key

# ======= 🤖 Inisialisasi Model LLM =======
def init_llm(api_key, model_name, temperature=0.7, max_tokens=500):
    """Inisialisasi model LLM"""
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,  # Corrected attribute name
            temperature=temperature,
            max_tokens=max_tokens
        )
        return llm
    except Exception as e:
        logger.error(f"Error inisialisasi LLM: {str(e)}")
        st.error(f"❌ Gagal menginisialisasi model AI: {str(e)}")
        return None

# ======= 🧠 Inisialisasi Memory =======
def init_memory(llm, max_token_limit=3000):  # Pass llm as argument
    """Inisialisasi memory dengan batasan token"""
    return ConversationTokenBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=max_token_limit,
        llm=llm  # Use the passed llm
    )

# ======= 📄 Proses File =======
def process_file(uploaded_file, chunk_size=500, chunk_overlap=50):
    """Proses file yang diunggah menjadi vector store"""
    try:
        with st.spinner("📖 Memproses file..."):
            start_time = time.time()

            # Menyimpan file dengan nama yang aman
            file_path = get_secure_temp_file(uploaded_file)

            # Mendeteksi tipe file dan memilih loader yang sesuai
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(file_path)
            elif uploaded_file.name.endswith('.csv'):
                loader = CSVLoader(file_path)
            else:
                st.error("❌ Format file tidak didukung. Silakan unggah file PDF, TXT, atau CSV.")
                return None

            # Load dokumen
            documents = loader.load()

            # Hitung jumlah halaman/baris
            doc_count = len(documents)

            # Gunakan RecursiveCharacterTextSplitter untuk pemisahan teks yang lebih baik
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )

            # Split dokumen
            split_docs = text_splitter.split_documents(documents)

            # Hitung jumlah chunks
            chunk_count = len(split_docs)

            # Buat embedding dan simpan ke FAISS
            with get_openai_callback() as cb:
                embeddings = OpenAIEmbeddings(api_key=st.session_state.api_key)
                vectorstore = FAISS.from_documents(split_docs, embeddings)

                # Update token usage
                st.session_state.token_usage["prompt_tokens"] += cb.prompt_tokens
                st.session_state.token_usage["completion_tokens"] += cb.completion_tokens
                st.session_state.token_usage["total_tokens"] += cb.total_tokens

            # Simpan informasi file
            st.session_state.file_info = {
                "filename": uploaded_file.name,
                "size_mb": round(uploaded_file.size / (1024 * 1024), 2),
                "doc_count": doc_count,
                "chunk_count": chunk_count,
                "processing_time": round(time.time() - start_time, 2)
            }

            # Kembalikan retriever
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

    except Exception as e:
        logger.error(f"Error memproses file: {str(e)}")
        st.error(f"❌ Gagal memproses file: {str(e)}")
        return None

# ======= 🔄 Inisialisasi Chain =======
def init_chain(retriever):
    """Inisialisasi ConversationalRetrievalChain"""
    # Template prompt kustom
    template = """
    Kamu adalah AI Business Consultant yang profesional dan membantu.

    Konten berikut adalah informasi yang relevan yang ditemukan dari dokumen yang diunggah:
    {context}

    Riwayat Percakapan:
    {chat_history}

    Pertanyaan Pengguna: {question}

    Berikan jawaban yang komprehensif, akurat, dan bermanfaat berdasarkan informasi yang diberikan.
    Jika jawaban tidak ditemukan dalam informasi yang tersedia, katakan dengan jujur bahwa 
    kamu tidak dapat menjawab berdasarkan dokumen yang diunggah.
    """

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=retriever,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        return chain
    except Exception as e:
        logger.error(f"Error inisialisasi chain: {str(e)}")
        st.error(f"❌ Gagal menginisialisasi chain: {str(e)}")
        return None

# ======= 🔍 Proses Internet Search =======
def search_internet(query):
    """Simulasi pencarian internet (implementasi sebenarnya membutuhkan API)"""
    st.info("🔍 Mencari di internet...")
    time.sleep(2)  # Simulasi delay pencarian

    # Implementasi pencarian sebenarnya akan ditambahkan di sini

    return f"Berikut adalah hasil pencarian untuk '{query}':\n\n" + \
           "1. Hasil pencarian yang relevan akan ditampilkan di sini.\n" + \
           "2. Informasi tambahan dari pencarian web akan diintergrasikan.\n" + \
           "3. Pencarian khusus untuk informasi bisnis akan diprioritaskan."

# ======= 🚀 Inisialisasi Aplikasi =======
initialize_app()

# ======= 📊 Sidebar =======
with st.sidebar:
    st.image("https://www.svgrepo.com/show/373328/ai.svg", width=100)
    st.title("💼 AI Business Consultant")

    # Load API Key
    st.session_state.api_key = load_api_key()

    # Model Selection
    model_options = {
        "gpt-4": "GPT-4 (Powerful & Accurate)",
        "gpt-3.5-turbo": "GPT-3.5 Turbo (Fast & Efficient)",
        "gpt-3.5-turbo-16k": "GPT-3.5 Turbo 16K (Extended Context)"
    }
    selected_model = st.selectbox(
        "🤖 Pilih Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )

    # Advanced Settings
    with st.expander("⚙️ Pengaturan Lanjutan"):
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens Response:", 256, 4096, 1024, 128)
        chunk_size = st.slider("Chunk Size:", 100, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap:", 0, 200, 50, 10)

    # Initialize LLM if API key is available AND (llm is not initialized OR model changed)
    if st.session_state.api_key and (st.session_state.llm is None or st.session_state.llm.model_name != selected_model):
        st.session_state.llm = init_llm(
            st.session_state.api_key,
            selected_model,
            temperature,
            max_tokens
        )
        # Reinitialize memory only when llm changes
        if st.session_state.llm: # check llm is not None before init memory
            st.session_state.memory = init_memory(st.session_state.llm) # Pass LLM



    # Token Usage Stats
    if st.session_state.token_usage["total_tokens"] > 0:
        st.write("📊 **Token Usage**")
        st.write(f"Total: {st.session_state.token_usage['total_tokens']}")
        st.write(f"Prompt: {st.session_state.token_usage['prompt_tokens']}")
        st.write(f"Completion: {st.session_state.token_usage['completion_tokens']}")

    # Control Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Chat", key="reset"):
            st.session_state.history = []
            if st.session_state.llm: # check llm is not None before init memory
                st.session_state.memory = init_memory(st.session_state.llm) # Pass LLM

            st.success("💡 Chat telah direset!")

    with col2:
        if st.button("🗑️ Clear Files", key="clear"):
            st.session_state.retriever = None
            st.session_state.file_processed = False
            st.session_state.file_info = {}
            cleanup_temp_files()
            st.success("🗑️ File telah dihapus!")

# ======= 🔹 Main Content =======
st.title("💼 AI Business Consultant")
st.write("Konsultan bisnis AI yang dapat membantu Anda dengan strategi bisnis
