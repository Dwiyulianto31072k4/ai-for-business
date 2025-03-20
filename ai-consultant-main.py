import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain  # Import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
import tempfile
#import speech_recognition as sr # Removed
from PIL import Image
from transformers import pipeline

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
system_template = """Anda adalah analis keuangan senior yang sangat terampil. Tugas Anda adalah menganalisis informasi keuangan yang diberikan, baik dari dokumen, teks, maupun gambar.

Informasi:
{context}

Gunakan Bahasa Indonesia formal dan istilah keuangan yang tepat. Jawablah dengan ringkas dan jelas."""

user_template = """{question}

{format_instructions}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

# Note: 'context' will be empty initially, then filled with document content if available.
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

# ======= 🎤 Speech-to-Text Function =======
# REMOVED transcribe_audio function

# ======= 🖼️ Image Captioning Function (Hugging Face) =======
# Moved initialization to main block

# ======= 🧩 Main Application Logic =======
def main():
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox("Select AI Model", ["gpt-4", "gpt-3.5-turbo"])
        analysis_depth = st.slider("Analysis Depth", 1, 5, 3)
        st.divider()

        if st.button("🧹 Clear Session", key='clear_session_button'): #Added key
            st.session_state.clear()
            st.cache_resource.clear()
            st.rerun()

    # Initialize Core Components
    models = init_models()
    llm = models[model_name]
    processor = FinancialDocumentProcessor()
    image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    # Initialize a basic LLMChain for conversation *before* document processing
    if "qa_chain" not in st.session_state:
        initial_chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)
        st.session_state.qa_chain = initial_chain

    st.title("📊 AI Financial Analyst Pro")
    st.caption("Enterprise-grade Financial Document Analysis System")

    # --- File Upload ---
    uploaded_files = st.file_uploader(
        "📁 Upload Financial Documents (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    # --- Image Upload ---
    uploaded_image = st.file_uploader("🖼️ Upload Image (optional)", type=["jpg", "jpeg", "png"])
    image_caption = ""  # Initialize image_caption
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        image_caption = image_to_text(uploaded_image)  # Generate caption


    # --- Document processing ---
    if uploaded_files: # Removed  and "retriever" not in st.session_state:
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
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 20}
                    )
                    # Create the ConversationalRetrievalChain *now*
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        memory=st.session_state.memory,
                        verbose=True,
                        return_source_documents=True,
                        combine_docs_chain_kwargs={"prompt": prompt}
                    )
                    st.session_state.qa_chain = qa_chain  # Update the qa_chain
                    st.session_state.retriever = retriever # Store the retriever

                status.update(label="✅ Analysis Complete", state="complete")
                st.success(f"Processed files: {', '.join(processed_files)}")
            else:
                status.update(label="⚠️ No valid documents processed", state="error")


    # --- Speech-to-Text ---
    # REMOVED Speech-to-text section

    # --- Chat Interface ---
    # Use st.session_state for user input (initialize if it doesn't exist)
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # st.chat_input *returns* the user's input.  It doesn't take a 'value' argument.
    prompt_text = st.chat_input("💬 Ask financial questions...", placeholder="Type here...")

    if prompt_text:
        st.session_state.user_input = prompt_text  # Update session state with the entered text

    # Use a consistent variable for the user's input (either transcribed or typed)
    user_query = st.session_state.user_input

    if user_query:  # Proceed if there's *any* input
        with st.chat_message("user"):
            st.markdown(user_query)

        # --- Dynamic Prompt ---
        format_instructions = ""
        if "analisis risiko" in user_query.lower():
            format_instructions += "Risk Analysis: (analisis risiko secara mendalam)\n"
        if "rekomendasi" in user_query.lower():
            format_instructions += "Recommendations: (rekomendasi yang actionable dan spesifik)\n"
        if not format_instructions:
            format_instructions = """
            Executive Summary: (maksimal 3 kalimat)
            Key Metrics: (dalam format tabel)
            """

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Combine text input and image caption
                    full_input = user_query
                    if image_caption:
                        full_input += "\n\nImage Caption: " + image_caption

                    # Use the appropriate chain (retrieval or basic)
                    if "retriever" in st.session_state:
                        # Retrieval chain (documents available)
                        result = st.session_state.qa_chain.invoke({"question": full_input, "format_instructions": format_instructions})
                        response = result["answer"]

                        if 'source_documents' in result:
                            with st.expander("Source Documents"):
                                for doc in result['source_documents']:
                                    st.write(doc.page_content)
                                    st.write(doc.metadata)
                                    st.write("---")
                    else:
                        # Basic LLMChain (no documents)
                         result = st.session_state.qa_chain.invoke({"question": full_input,"context":"","format_instructions":format_instructions})
                         response = result["text"]


                    # --- Post-processing (same as before, but using 'response') ---
                    parts = {}
                    if "Executive Summary" in response:
                        try:
                            parts["Executive Summary"] = response.split("Executive Summary")[1].split("Key Metrics")[0].strip()
                        except:
                            pass
                    if "Key Metrics" in response:
                        try:
                            table_str = response.split("Key Metrics")[1].split("Risk Analysis")[0].split("Recommendations")[0].strip()
                            rows = [line.split("|") for line in table_str.split("\n") if line.strip()]
                            header = [h.strip() for h in rows[0] if h.strip()]
                            data = [[d.strip() for d in row if d.strip()] for row in rows[1:] if any(row)]
                            if header and data:
                                parts["Key Metrics"] = (header, data)
                            else:
                                parts["Key Metrics"] = table_str
                        except Exception as e:
                            parts["Key Metrics"] = table_str

                    if "Risk Analysis" in response:
                        parts["Risk Analysis"] = response.split("Risk Analysis")[1].split("Recommendations")[0].strip()
                    if "Recommendations" in response:
                        parts["Recommendations"] = response.split("Recommendations")[1].strip()

                    if "Executive Summary" in parts:
                        st.markdown(parts["Executive Summary"])

                    if "Key Metrics" in parts:
                        if isinstance(parts["Key Metrics"], tuple):
                            st.table(data=parts["Key Metrics"][1], header=parts["Key Metrics"][0])
                        else:
                            st.markdown(parts["Key Metrics"])

                    if "Risk Analysis" in parts:
                        st.markdown("**Risk Analysis**")
                        st.markdown(parts["Risk Analysis"])

                    if "Recommendations" in parts:
                        st.markdown("**Recommendations**")
                        st.markdown(parts["Recommendations"])

                    if not parts:
                        st.markdown(response)


                except Exception as e:
                    st.error(f"⚠️ AI Response Error: {type(e).__name__}: {str(e)}")

        # Clear user_input *after* processing, so it's ready for the next turn.
        st.session_state.user_input = ""


# ======= 🚨 Error Handling & Safety =======
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Critical System Error: {type(e).__name__}: {str(e)}")
        st.error("Please refresh the browser and try again.")
