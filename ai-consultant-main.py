import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
import tempfile
import re

# ======= 🚀 Initial Configuration =======
st.set_page_config(
    page_title="AI Financial Analyst Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======= 🔐 Security & Authentication =======
if "OPENAI_API_KEY" not in st.secrets:
    st.error("🔒 API Key not found! Configure in Streamlit Secrets.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# ======= 🧠 Advanced Memory Management =======
class EnhancedConversationMemory:
    def __init__(self):
        self.history = []
        self.context_memory = []
        
    def add_context(self, context):
        self.context_memory.append(context)
        
    def get_relevant_context(self, query):
        return "\n".join([c for c in self.context_memory if query.lower() in c.lower()][-3:])

if "memory" not in st.session_state:
    st.session_state.memory = EnhancedConversationMemory()

# ======= 📈 Financial Analysis Templates =======
FINANCIAL_ANALYSIS_PROMPT = """Anda adalah analis keuangan senior. Analisis dokumen berikut:

{docs}

Pertanyaan: {question}

Format jawaban:
1. **Executive Summary** (maks 3 kalimat)
2. **Key Metrics** (format tabel)
3. **Risk Analysis**
4. **Recommendations**

Gunakan Bahasa Indonesia formal dan istilah keuangan yang tepat."""

# ======= 🛠️ Document Processing Engine =======
class FinancialDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def process_file(self, file_path, file_type):
        try:
            if file_type == "application/pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "text/plain":
                loader = TextLoader(file_path)
            else:
                raise ValueError("Unsupported file type")
                
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"⚠️ Document processing error: {str(e)}")
            return None

# ======= 🤖 AI Model Initialization =======
def init_models():
    return {
        "gpt-4": ChatOpenAI(temperature=0.2, model="gpt-4", api_key=openai_api_key),
        "gpt-3.5-turbo": ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo", api_key=openai_api_key)
    }

# ======= 🧩 Main Application Logic =======
def main():
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox("Select AI Model", ["gpt-4", "gpt-3.5-turbo"])
        analysis_depth = st.slider("Analysis Depth", 1, 5, 3)
        st.divider()
        
        if st.button("🧹 Clear Session"):
            st.session_state.clear()
            st.rerun()

    # Initialize Core Components
    models = init_models()
    llm = models[model_name]
    processor = FinancialDocumentProcessor()
    
    # Main Interface
    st.title("📊 AI Financial Analyst Pro")
    st.caption("Enterprise-grade Financial Document Analysis System")

    # File Upload Section
    uploaded_files = st.file_uploader(
        "📁 Upload Financial Documents (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    # Document Processing
    retriever = None
    if uploaded_files:
        with st.status("🔍 Analyzing Documents...", expanded=True) as status:
            all_docs = []
            for uploaded_file in uploaded_files:
                try:
                    st.write(f"Processing {uploaded_file.name}...")
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name
                    
                    split_docs = processor.process_file(temp_path, uploaded_file.type)
                    if split_docs:
                        all_docs.extend(split_docs)
                        st.session_state.memory.add_context(
                            f"Document Processed: {uploaded_file.name}"
                        )
                    
                    os.unlink(temp_path)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            if all_docs:
                with st.spinner("🧠 Building Knowledge Base..."):
                    vectorstore = FAISS.from_documents(all_docs, OpenAIEmbeddings())
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 20}
                    )
                status.update(label="✅ Analysis Complete", state="complete")
            else:
                st.error("⚠️ No valid documents processed")

    # Chat Interface
    if prompt := st.chat_input("💬 Ask financial questions..."):
        # Display User Message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare Context
        context = ""
        if retriever:
            docs = retriever.get_relevant_documents(prompt)
            context = "\n\n".join([d.page_content for d in docs][:5])
        
        # Enhanced Prompt Engineering
        custom_prompt = PromptTemplate.from_template(FINANCIAL_ANALYSIS_PROMPT)
        formatted_prompt = custom_prompt.format(
            docs=context,
            question=prompt
        )

        # Generate Response
        with st.chat_message("assistant"):
            try:
                response = llm.invoke(formatted_prompt)
                content = response.content
                
                # Enhanced Formatting
                content = re.sub(r"\*\*(.*?)\*\*", r"**\1**", content)  # Bold formatting
                content = re.sub(r"(\d+\.)\s", r"\n\\1 ", content)       # List formatting
                
                st.markdown(content)
                st.session_state.memory.add_context(f"Q: {prompt}\nA: {content}")
            except Exception as e:
                st.error(f"⚠️ AI Response Error: {str(e)}")

# ======= 🚨 Error Handling & Safety =======
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Critical System Error: {str(e)}")
        st.error("Please refresh the browser and try again")
