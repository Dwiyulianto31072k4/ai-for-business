import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
import tempfile

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

# ======= 🧠 Memory Management (Simplified with LangChain) =======
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ======= 📈 Financial Analysis Templates (Dynamic Prompting) =======

system_template = """Anda adalah analis keuangan senior yang sangat terampil. Tugas Anda adalah menganalisis dokumen keuangan yang diberikan.

Dokumen:
{context}

Gunakan Bahasa Indonesia formal dan istilah keuangan yang tepat. Jawablah dengan ringkas dan jelas."""

user_template = """{question}

{format_instructions}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]
prompt = PromptTemplate(input_variables=["context", "question", "format_instructions"], template=system_template + "\n\n" + user_template)


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
                raise ValueError(f"Unsupported file type: {file_type}")

            documents = loader.load()
            return self.text_splitter.split_documents(documents)

        except FileNotFoundError:
            st.error(f"⚠️ File not found: {file_path}")
            return None
        except ValueError as ve:
            st.error(f"⚠️ Value error: {ve}")
            return None
        except Exception as e:
            st.error(f"⚠️ Document processing error: {type(e).__name__}: {str(e)}")
            return None

# ======= 🤖 AI Model Initialization =======
def init_models():
    return {
        "gpt-4": ChatOpenAI(temperature=0.2, model="gpt-4", api_key=openai_api_key, streaming=True),
        "gpt-3.5-turbo": ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo", api_key=openai_api_key, streaming=True)
    }

# ======= 🧩 Main Application Logic =======
def main():
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox("Select AI Model", ["gpt-4", "gpt-3.5-turbo"])
        analysis_depth = st.slider("Analysis Depth", 1, 5, 3)  # Currently unused, but good for future features
        st.divider()

        if st.button("🧹 Clear Session"):
            st.session_state.clear()
            st.cache_resource.clear()
            st.rerun()

    # Initialize Core Components
    models = init_models()
    llm = models[model_name]
    processor = FinancialDocumentProcessor()

    st.title("📊 AI Financial Analyst Pro")
    st.caption("Enterprise-grade Financial Document Analysis System")

    uploaded_files = st.file_uploader(
        "📁 Upload Financial Documents (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    # Document Processing and Retriever (only if not already in session_state)
    if uploaded_files and "retriever" not in st.session_state:
        with st.status("🔍 Analyzing Documents...", expanded=True) as status:
            all_docs = []
            processed_files = []

            for uploaded_file in uploaded_files:
                try:
                    st.write(f"Processing {uploaded_file.name}...")
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    split_docs = processor.process_file(temp_path, uploaded_file.type)
                    if split_docs:
                        all_docs.extend(split_docs)
                        processed_files.append(uploaded_file.name)
                    os.unlink(temp_path)

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

            if all_docs:
                with st.spinner("🧠 Building Knowledge Base..."):
                    vectorstore = FAISS.from_documents(all_docs, OpenAIEmbeddings(api_key=openai_api_key))
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 20}
                    )
                status.update(label="✅ Analysis Complete", state="complete")
                st.success(f"Processed files: {', '.join(processed_files)}")
            else:
                status.update(label="⚠️ No valid documents processed", state="error")


    # Chat Interface
    if prompt := st.chat_input("💬 Ask financial questions..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        if "retriever" not in st.session_state:
            st.error("Please upload documents first.")
            st.stop()

        retriever = st.session_state.retriever

        # --- Dynamic Formatting Logic ---
        format_instructions = ""
        if "analisis risiko" in prompt.lower():
            format_instructions += "Risk Analysis: (analisis risiko secara mendalam)\n"  # No numbering
        if "rekomendasi" in prompt.lower():
            format_instructions += "Recommendations: (rekomendasi yang actionable dan spesifik)\n"  # No numbering

        if not format_instructions:  # Default format if no keywords
            format_instructions = """
            Executive Summary: (maksimal 3 kalimat)
            Key Metrics: (dalam format tabel)
            """

        # --- End Dynamic Logic ---

        # Perhatikan perubahan di sini:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            verbose=True,
            # combine_docs_chain_kwargs={"prompt": prompt}  <-- HAPUS INI
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
              try:
                    # Perhatikan perubahan di sini juga:
                    result = qa_chain.invoke({"question": prompt, "format_instructions": format_instructions, "context": ""}) #context dikosongkan, karena sudah di handle oleh retriever
                    response = result["answer"]


                    # --- Improved Post-processing ---
                    parts = {}  # Dictionary to store extracted parts

                    # 1. Executive Summary
                    if "Executive Summary" in response:
                        try:
                            parts["Executive Summary"] = response.split("Executive Summary")[1].split("Key Metrics")[0].strip()
                        except:
                            pass

                    # 2. Key Metrics
                    if "Key Metrics" in response:
                        try:
                            table_str = response.split("Key Metrics")[1].split("Risk Analysis")[0].split("Recommendations")[0].strip()
                            rows = [line.split("|") for line in table_str.split("\n") if line.strip()]
                            header = [h.strip() for h in rows[0] if h.strip()]
                            data = [[d.strip() for d in row if d.strip()] for row in rows[1:] if any(row)]
                            if header and data:
                                parts["Key Metrics"] = (header, data) # Store header and data separately
                            else:
                                parts["Key Metrics"] = table_str # Fallback: Store raw string
                        except Exception as e:
                          parts["Key Metrics"] = table_str

                    # 3. & 4. Risk Analysis and Recommendations
                    if "Risk Analysis" in response:
                         parts["Risk Analysis"] = response.split("Risk Analysis")[1].split("Recommendations")[0].strip()
                    if "Recommendations" in response:
                         parts["Recommendations"] = response.split("Recommendations")[1].strip()



                    # --- Display Logic ---
                    if "Executive Summary" in parts:
                        st.markdown(parts["Executive Summary"]) #just show it

                    if "Key Metrics" in parts:
                        if isinstance(parts["Key Metrics"], tuple): # It's (header, data)
                            st.table(data=parts["Key Metrics"][1], header=parts["Key Metrics"][0]) #use st.table
                        else:
                            st.markdown(parts["Key Metrics"]) # Fallback: raw text

                    if "Risk Analysis" in parts:
                        st.markdown("**Risk Analysis**") # Add a heading
                        st.markdown(parts["Risk Analysis"])

                    if "Recommendations" in parts:
                        st.markdown("**Recommendations**")  # Add a heading
                        st.markdown(parts["Recommendations"])


                    # Fallback: Display entire response if nothing was extracted
                    if not parts:
                        st.markdown(response)

              except Exception as e:
                    st.error(f"⚠️ AI Response Error: {type(e).__name__}: {str(e)}")


# ======= 🚨 Error Handling & Safety =======
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Critical System Error: {type(e).__name__}: {str(e)}")
        st.error("Please refresh the browser and try again.")
