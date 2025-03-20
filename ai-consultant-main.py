import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
import os
import uuid
import time
import threading
import hashlib
import pandas as pd
import logging
import tempfile
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime

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
def sanitize_filename(filename: str) -> str:
    """Sanitasi nama file untuk keamanan"""
    return hashlib.md5(filename.encode()).hexdigest()

def get_secure_temp_file(uploaded_file) -> str:
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
    state_vars = {
        "conversation": None,
        "retriever": None,
        "history": [],
        "file_processed": False,
        "file_info": {},
        "token_usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
        "llm": None,
        "memory": None,
        "current_model": "gpt-4",
        "chain": None,
        "temp_files": [],
        "api_key": None,
        "session_id": str(uuid.uuid4()),
        "last_activity": datetime.now().timestamp()
    }
    
    for var, default in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

# ======= 🔍 Load API Key =======
def load_api_key() -> str:
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
def init_llm(api_key: str, model_name: str, temperature: float = 0.7, max_tokens: int = 500):
    """Inisialisasi model LLM"""
    try:
        if not api_key:
            return None
            
        llm = ChatOpenAI(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Simpan model yang sedang digunakan dalam state
        st.session_state.current_model = model_name
        return llm
    except Exception as e:
        logger.error(f"Error inisialisasi LLM: {str(e)}")
        st.error(f"❌ Gagal menginisialisasi model AI: {str(e)}")
        return None

# ======= 🧠 Inisialisasi Memory =======
def init_memory(max_token_limit: int = 3000):
    """Inisialisasi memory dengan batasan token"""
    if not st.session_state.llm:
        return None
        
    return ConversationTokenBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",  # Tetapkan output_key untuk mengatasi error
        max_token_limit=max_token_limit,
        llm=st.session_state.llm
    )

# ======= 📄 Proses File =======
def process_file(uploaded_file, chunk_size: int = 500, chunk_overlap: int = 50):
    """Proses file yang diunggah menjadi vector store"""
    try:
        with st.spinner("📖 Memproses file..."):
            start_time = time.time()
            
            # Menyimpan file dengan nama yang aman
            file_path = get_secure_temp_file(uploaded_file)
            
            # Mendeteksi tipe file dan memilih loader yang sesuai
            if uploaded_file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.lower().endswith('.txt'):
                loader = TextLoader(file_path)
            elif uploaded_file.name.lower().endswith('.csv'):
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
    if not st.session_state.llm or not st.session_state.memory:
        return None
        
    template = """
    Kamu adalah AI Business Consultant yang profesional, cerdas, dan membantu.
    
    Nama kamu adalah "Business AI Pro" dan kamu memiliki pengalaman luas di bidang konsultasi bisnis.
    
    Konten berikut adalah informasi yang relevan yang ditemukan dari dokumen yang diunggah:
    {context}
    
    Riwayat Percakapan:
    {chat_history}
    
    Pertanyaan Pengguna: {question}
    
    Berikan jawaban yang komprehensif, akurat, dan bermanfaat berdasarkan informasi yang diberikan.
    Jika jawaban tidak ditemukan dalam informasi yang tersedia, katakan dengan jujur bahwa
    kamu tidak dapat menjawab berdasarkan dokumen yang diunggah.
    
    Berikan format yang rapi dengan poin-poin dan penekanan pada bagian penting.
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
            return_source_documents=True,
            verbose=True
        )

        return chain

    except Exception as e:
        logger.error(f"Error inisialisasi chain: {str(e)}")
        st.error(f"❌ Gagal menginisialisasi chain: {str(e)}")
        return None

# ======= 🔍 Proses Internet Search =======
def search_internet(query: str) -> str:
    """Simulasi pencarian internet (implementasi sebenarnya membutuhkan API)"""
    st.info("🔍 Mencari di internet...")
    time.sleep(2)  # Simulasi delay pencarian
    
    # Di sini Anda bisa mengimplementasikan pencarian web sebenarnya
    # Contoh: Menggunakan Google Search API, Bing API, dll.
    
    return f"Berikut adalah hasil pencarian untuk '{query}':\n\n" + \
           "1. Hasil pencarian yang relevan akan ditampilkan di sini.\n" + \
           "2. Informasi tambahan dari pencarian web akan diintergrasikan.\n" + \
           "3. Pencarian khusus untuk informasi bisnis akan diprioritaskan."

# ======= 💬 Proses Query =======
def process_query(query: str) -> Tuple[str, List]:
    """Memproses kueri dan menangani output."""
    try:
        # Perbarui timestamp aktivitas terakhir
        st.session_state.last_activity = datetime.now().timestamp()
        
        # Deteksi jika ada permintaan pencarian web
        if any(keyword in query.lower() for keyword in ["cari di internet", "search online", "cari online"]):
            return search_internet(query), []
        
        # Jika ada file yang diproses, gunakan ConversationalRetrievalChain
        if st.session_state.file_processed and st.session_state.conversation:
            with get_openai_callback() as cb:
                result = st.session_state.conversation({"question": query})
                
                # Update token usage
                st.session_state.token_usage["prompt_tokens"] += cb.prompt_tokens
                st.session_state.token_usage["completion_tokens"] += cb.completion_tokens
                st.session_state.token_usage["total_tokens"] += cb.total_tokens
                
                response = result["answer"]
                source_docs = result.get("source_documents", [])
                
                # Format respons dengan Markdown untuk meningkatkan keterbacaan
                response = format_response(response)
                
                # Tambahkan informasi sumber
                if source_docs:
                    sources = set()
                    for doc in source_docs:
                        if hasattr(doc, "metadata") and "source" in doc.metadata:
                            sources.add(doc.metadata["source"])
                    
                    if sources:
                        response += "\n\n**Sumber:**\n"
                        for source in sources:
                            response += f"- {source}\n"
                
                return response, source_docs
                
        # Jika tidak ada file, gunakan LLM langsung
        else:
            with get_openai_callback() as cb:
                prompt = f"""
                Kamu adalah AI Business Consultant Pro yang profesional dan membantu.
                
                Nama kamu adalah "Business AI Pro" dan kamu memiliki pengalaman luas di bidang strategi bisnis,
                pemasaran, keuangan, manajemen, dan pengembangan produk.
                
                Pertanyaan pengguna: {query}
                
                Berikan jawaban yang komprehensif, terstruktur, dan bermanfaat.
                Format respons menggunakan Markdown untuk meningkatkan keterbacaan.
                """
                
                result = st.session_state.llm.invoke(prompt)
                
                # Update token usage
                st.session_state.token_usage["prompt_tokens"] += cb.prompt_tokens
                st.session_state.token_usage["completion_tokens"] += cb.completion_tokens
                st.session_state.token_usage["total_tokens"] += cb.total_tokens
                
                # Format respons
                response = format_response(result.content)
                
                return response, []
                
    except Exception as e:
        logger.error(f"Error saat memproses pertanyaan: {str(e)}")
        return f"⚠️ Terjadi kesalahan: {str(e)}", []

# ======= 🎨 Format Response =======
def format_response(text: str) -> str:
    """Memformat respons untuk tampilan yang lebih baik"""
    # Pastikan heading memiliki spasi setelah tanda #
    text = re.sub(r'(#{1,6})([^ #])', r'\1 \2', text)
    
    # Tambahkan penekanan pada poin-poin penting jika belum ada
    if "**" not in text and "*" not in text:
        # Cari frasa-frasa penting untuk diberi penekanan
        important_phrases = ["penting", "kunci", "utama", "strategi", "rekomendasi", "solusi"]
        for phrase in important_phrases:
            pattern = re.compile(r'(\w*' + phrase + r'\w*)', re.IGNORECASE)
            text = pattern.sub(r'**\1**', text)
    
    return text

# ======= 🚀 Inisialisasi Aplikasi =======
initialize_app()

# ======= 📊 Sidebar =======
with st.sidebar:
    st.image("https://www.provincial.com/content/dam/public-web/global/images/micro-illustrations/bbva_manager_man_2.im1705594061549im.png?imwidth=320", width=100)
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
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.current_model)
    )
    
    # Advanced Settings
    with st.expander("⚙️ Pengaturan Lanjutan"):
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1, 
                               help="Nilai rendah: lebih fokus dan deterministik. Nilai tinggi: lebih kreatif dan bervariasi.")
        
        max_tokens = st.slider("Max Tokens Response:", 256, 4096, 1024, 128,
                              help="Batas maksimum token untuk respons AI.")
        
        chunk_size = st.slider("Chunk Size:", 100, 1000, 500, 50,
                             help="Ukuran potongan teks untuk pemrosesan dokumen.")
        
        chunk_overlap = st.slider("Chunk Overlap:", 0, 200, 50, 10,
                                help="Jumlah overlap antar potongan teks untuk mempertahankan konteks.")
    
    # Initialize LLM if API key is available
    if st.session_state.api_key and (st.session_state.llm is None or 
                                    selected_model != st.session_state.current_model):
        st.session_state.llm = init_llm(
            st.session_state.api_key,
            selected_model,
            temperature,
            max_tokens
        )
        
        # Reinitialize memory with new LLM
        if st.session_state.llm:
            st.session_state.memory = init_memory()
    
    # Token Usage Stats
    if st.session_state.token_usage["total_tokens"] > 0:
        st.write("📊 **Token Usage**")
        col1, col2 = st.columns(2)
        col1.metric("Total", f"{st.session_state.token_usage['total_tokens']:,}")
        col2.metric("Prompt", f"{st.session_state.token_usage['prompt_tokens']:,}")
        st.metric("Completion", f"{st.session_state.token_usage['completion_tokens']:,}")
        
        # Estimasi biaya (contoh harga, perlu disesuaikan)
        if st.session_state.current_model == "gpt-4":
            prompt_cost = st.session_state.token_usage['prompt_tokens'] * 0.00003
            completion_cost = st.session_state.token_usage['completion_tokens'] * 0.00006
        else:  # gpt-3.5-turbo
            prompt_cost = st.session_state.token_usage['prompt_tokens'] * 0.0000015
            completion_cost = st.session_state.token_usage['completion_tokens'] * 0.000002
            
        total_cost = prompt_cost + completion_cost
        st.write(f"💰 **Estimasi Biaya:** ${total_cost:.4f}")
    
    # Control Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Chat", key="reset"):
            st.session_state.history = []
            st.session_state.memory = init_memory()
            st.success("💡 Chat telah direset!")
    
    with col2:
        if st.button("🗑️ Clear Files", key="clear"):
            st.session_state.retriever = None
            st.session_state.file_processed = False
            st.session_state.file_info = {}
            cleanup_temp_files()
            st.success("🗑️ File telah dihapus!")

# ======= 🔹 Main Content =======
st.title("💼 AI Business Consultant Pro")
st.markdown("""
Konsultan bisnis AI yang dapat membantu Anda dengan strategi bisnis, 
analisis data, pemasaran, keuangan, dan rekomendasi berdasarkan dokumen yang Anda unggah.
""")

# Tabs for different modes
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📂 Upload File", "ℹ️ Bantuan"])

# Help Tab
with tab3:
    st.subheader("📘 Panduan Penggunaan")
    
    st.markdown("""
    ### 🌟 Cara Menggunakan AI Business Consultant Pro

    #### 1️⃣ Persiapan
    - Masukkan API Key OpenAI di sidebar (jika belum dikonfigurasi)
    - Pilih model AI yang ingin digunakan (GPT-4 direkomendasikan untuk hasil terbaik)
    
    #### 2️⃣ Upload Dokumen (Opsional)
    - Unggah dokumen bisnis Anda (PDF, TXT, CSV)
    - Klik tombol "Proses File" untuk menganalisis dokumen
    - Dokumen akan dipecah dan dikonversi menjadi basis pengetahuan
    
    #### 3️⃣ Konsultasi dengan AI
    - Ajukan pertanyaan tentang bisnis Anda
    - AI akan menjawab berdasarkan pengetahuan umum atau dokumen yang diunggah
    - Untuk pencarian internet, tambahkan frasa "cari di internet" dalam pertanyaan Anda
    
    #### 📋 Contoh Pertanyaan untuk AI:
    - "Bagaimana strategi pemasaran yang efektif untuk startup teknologi?"
    - "Analisis SWOT untuk bisnis retail berdasarkan data yang saya unggah"
    - "Buat rencana bisnis untuk perusahaan jasa konsultasi"
    - "Cari di internet tentang tren e-commerce terbaru"
    """)
    
    st.info("💡 **Tip**: Untuk hasil terbaik, unggah dokumen yang berisi informasi spesifik tentang bisnis Anda.")
    
    # Contoh format dokumen
    with st.expander("📄 Format Dokumen yang Didukung"):
        st.markdown("""
        ### Format Dokumen:
        
        1. **PDF (.pdf)**
           - Laporan keuangan
           - Rencana bisnis
           - Studi kasus
           - Analisis pasar
           
        2. **Text (.txt)**
           - Data mentah
           - Catatan rapat
           - Deskripsi produk
           
        3. **CSV (.csv)**
           - Data penjualan
           - Metrik kinerja
           - Data pelanggan
           - Analisis keuangan
        """)

# Upload File Tab
with tab2:
    st.subheader("📂 Unggah Dokumen")
    
    uploaded_file = st.file_uploader(
        "📎 Unggah file (PDF, TXT, CSV) untuk analisis AI", 
        type=["pdf", "txt", "csv"],
        accept_multiple_files=False,
        help="Unggah dokumen untuk dianalisis oleh AI. Dokumen akan dipecah dan diubah menjadi basis pengetahuan."
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        process_button = st.button("✅ Proses File", type="primary")
    
    with col2:
        if uploaded_file:
            st.write(f"File dipilih: **{uploaded_file.name}**")
    
    if process_button and uploaded_file:
        # Proses file
        with st.spinner("🔍 Menganalisis dokumen... Mohon tunggu"):
            st.session_state.retriever = process_file(
                uploaded_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if st.session_state.retriever:
                st.session_state.file_processed = True
                # Initialize conversation chain
                st.session_state.conversation = init_chain(st.session_state.retriever)
                st.success("✅ Dokumen berhasil diproses dan siap digunakan untuk konsultasi!")
    
    # Tampilkan informasi file jika sudah diproses
    if st.session_state.file_processed and st.session_state.file_info:
        st.subheader("📊 Informasi Dokumen")
        info = st.session_state.file_info
        
        # Metric cards dalam 3 kolom
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Nama File", info["filename"])
            st.metric("📏 Ukuran", f"{info['size_mb']} MB")
        
        with col2:
            st.metric("📚 Jumlah Halaman/Baris", info["doc_count"])
            st.metric("🧩 Jumlah Chunks", info["chunk_count"])
            
        with col3:
            st.metric("⏱️ Waktu Proses", f"{info['processing_time']} detik")
            estimated_queries = info["chunk_count"] * 2
            st.metric("🔍 Estimasi Kapasitas Query", f"~{estimated_queries} pertanyaan")
            
        st.info("💡 Dokumen Anda telah dikonversi menjadi basis pengetahuan AI. Anda sekarang dapat mengajukan pertanyaan tentang isinya di tab Chat.")

# Chat Tab
with tab1:
    st.subheader("💬 Chat dengan AI Business Consultant")
    
    # Tampilkan status koneksi file
    if st.session_state.file_processed:
        st.success(f"📄 Dokumen terhubung: **{st.session_state.file_info['filename']}**")
        st.markdown("AI akan menjawab berdasarkan dokumen yang diunggah dan pengetahuan umumnya.")
    else:
        st.info("💡 AI akan menjawab berdasarkan pengetahuan umumnya. Unggah dokumen di tab 'Upload File' untuk mendapatkan jawaban yang spesifik.")
    
    # Chat container
    chat_container = st.container()
    
    # Tampilkan chat history
    with chat_container:
        for i, (role, message) in enumerate(st.session_state.history):
            message_key = f"{role}_{i}"
            with st.chat_message(role, key=message_key):
                st.markdown(message)
    
    # Disable chat input jika API key tidak tersedia
    if not st.session_state.api_key:
        st.warning("⚠️ Silakan masukkan API Key OpenAI di sidebar untuk memulai chat.")
        chat_input_disabled = True
    else:
        chat_input_disabled = False
    
    # Chat input
    user_input = st.chat_input("✏️ Ketik pesan atau pertanyaan Anda...", disabled=chat_input_disabled)
    
    if user_input:
        # Tampilkan pesan user
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Tambahkan ke history
        st.session_state.history.append(("user", user_input))
        
        # Proses pertanyaan
        with st.spinner("🧠 AI sedang menganalisis..."):
            response, source_docs = process_query(user_input)
            
            # Tambahkan respons ke history
            st.session_state.history.append(("assistant", response))
            
            # Tampilkan respons
            with st.chat_message("assistant"):
                st.markdown(response)

# ======= 🎨 Custom CSS =======

# ======= 🎨 Custom CSS =======
st.markdown("""
<style>
    /* Gaya umum */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Gaya untuk container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Gaya untuk judul */
    h1, h2, h3 {
        font-weight: 600 !important;
    }
    
    /* Gaya untuk chat messages */
    .stChatMessage {
        border-radius: 15px !important;
        padding: 12px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatMessage:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.08) !important;
    }
    
    /* Gaya untuk pesan user */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #E7F3FE !important;
        border-left: 5px solid #4285F4 !important;
    }
    
    /* Gaya untuk pesan assistant */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #F1F8F5 !important;
        border-left: 5px solid #34A853 !important;
    }
    
    /* Gaya untuk chat input */
    .stChatInput {
        border-radius: 20px !important;
        border: 1px solid #DFE1E5 !important;
        padding: 12px !important;
        background-color: #F8F9FA !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInput:focus {
        border-color: #4285F4 !important;
        box-shadow: 0 0 8px rgba(66, 133, 244, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)



