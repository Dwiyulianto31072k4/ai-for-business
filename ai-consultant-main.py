import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# ======= 🚀 Konfigurasi Streamlit =======
st.set_page_config(page_title="AI Business Consultant", layout="wide")

# ======= 🚀 Load API Key dari Streamlit Secrets =======
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ API Key OpenAI tidak ditemukan! Tambahkan di Streamlit Cloud.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# ======= 💾 Simpan history chat di session_state =======
if "history" not in st.session_state:
    st.session_state.history = []

# ======= 🚀 Inisialisasi Chatbot =======
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")

# ======= 📌 Inisialisasi Memory untuk Chat History =======
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ======= 🔹 Pilihan Mode Chat atau Upload File =======
mode = st.radio("📌 Pilih Mode Interaksi:", ["Chat", "Upload File"], horizontal=True)

retriever = None  # Placeholder untuk retriever

if mode == "Upload File":
    uploaded_file = st.file_uploader("📎 Unggah file (PDF, TXT)", type=["pdf", "txt"])
    
    if uploaded_file:
        with st.spinner("📖 Memproses file..."):
            file_path = f"./temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load file sesuai jenisnya
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(file_path)
            elif uploaded_file.type == "text/plain":
                loader = TextLoader(file_path)

            # Proses teks dan simpan ke retriever
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = text_splitter.split_documents(documents)
            retriever = FAISS.from_documents(split_docs, OpenAIEmbeddings()).as_retriever()

            st.success("✅ File berhasil diunggah dan diproses!")

# ======= 🔹 Chatbot dengan Memory & Knowledge dari File (Jika Ada) =======
conversation = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory
) if retriever else None

# ======= 💬 Tampilkan History Chat =======
st.markdown("## 💬 AI Business Consultant")
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

# ======= 🔹 Input Chat =======
user_input = st.chat_input("✏️ Ketik pesan Anda...")

if user_input:
    user_input = user_input.strip()

    # Simpan pertanyaan user ke chat history
    st.session_state.history.append(("user", user_input))

    with st.chat_message("user"):
        st.write(user_input)

    # ======= 🔥 LOGIKA PERBAIKAN MEMORY =======
    if retriever:
        try:
            response_data = conversation.invoke({"question": user_input})
            response = response_data.get("answer", "⚠️ Tidak ada jawaban yang tersedia.")
        except Exception as e:
            response = f"⚠️ Terjadi kesalahan dalam pemrosesan file: {str(e)}"
    else:
        try:
            response_data = llm.invoke(user_input)

            # ======= 🎯 PERBAIKAN MEMORY CHAT =======
            if isinstance(response_data, str):
                response = response_data  # Jika langsung string
            elif hasattr(response_data, "content"):
                response = response_data.content  # Jika objek AIMessage
            else:
                response = "⚠️ Tidak ada jawaban yang tersedia."

            # **Tambahkan hasil AI ke memory chat**
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response)

        except Exception as e:
            response = f"⚠️ Terjadi kesalahan dalam memproses pertanyaan: {str(e)}"

    # 💡 Pastikan tidak ada respons kosong
    if not response or not response.strip():
        response = "⚠️ Terjadi kesalahan dalam mendapatkan jawaban."

    # Simpan respons chatbot ke chat history
    st.session_state.history.append(("assistant", response))

    # ✅ Tampilkan respons chatbot di UI
    with st.chat_message("assistant"):
        st.write(response)

# ======= 🛠️ Fungsi Reset Chat =======
def reset_chat():
    st.session_state.history = []
    memory.clear()
    st.success("💡 Chat telah direset!")

st.sidebar.button("🔄 Reset Chat", on_click=reset_chat)

# ======= 🎨 UI CUSTOM =======
st.markdown("""
<style>
    .stChatMessage { border-radius: 8px; padding: 10px; margin: 5px; }
    .stChatMessage-user { background-color: #DCF8C6; color: black; text-align: right; }
    .stChatMessage-assistant { background-color: #ECECEC; color: black; text-align: left; }
    .stChatInput { border-radius: 20px; border: 1px solid #ccc; padding: 10px; width: 100%; background-color: #FAFAFA; }
    .stFileUploader { border: 2px dashed #ccc; border-radius: 10px; padding: 15px; text-align: center; font-size: 14px; color: #555; }
</style>
""", unsafe_allow_html=True)
