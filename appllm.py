import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
import pytesseract
from PIL import Image


st.set_page_config(page_title="Page Title", layout="wide")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
    """, unsafe_allow_html=True)

# Set USER_AGENT environment variable
os.environ['USER_AGENT'] = 'MyUserAgent'

# Initialize the Ollama model
model_local = ChatOllama(model='deepseek-r1')

# Streamlit App
st.sidebar.title("Include context")

# Sidebar for input type selection
input_type = st.sidebar.selectbox("Select Input Type", ["URLs", "Files", "Images"])

retriever = None
documents = []

if input_type == "URLs":
    # URL Input
    st.sidebar.subheader("Enter URLs")
    urls = st.sidebar.text_area(
        "Provide URLs (one per line):",
        value="",
    ).splitlines()
    if urls:
        try:
            # Load documents from URLs
            docs = [WebBaseLoader(url.strip()).load() for url in urls if url.strip()]
            docs_list = [item for sublist in docs for item in sublist]
            documents.extend(docs_list)
            st.sidebar.success("URLs processed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error processing URLs: {e}")

elif input_type == "Files":
    # File Upload
    st.sidebar.subheader("Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload text, PDF, or Word documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith(".txt"):
                    content = uploaded_file.read().decode("utf-8")
                    documents.append({"content": content, "metadata": {"source": uploaded_file.name}})
                elif uploaded_file.name.endswith(".pdf"):
                    from PyPDF2 import PdfReader
                    reader = PdfReader(uploaded_file)
                    content = "".join([page.extract_text() for page in reader.pages])
                    documents.append({"content": content, "metadata": {"source": uploaded_file.name}})
                elif uploaded_file.name.endswith(".docx"):
                    from docx import Document
                    doc = Document(uploaded_file)
                    content = "\n".join([p.text for p in doc.paragraphs])
                    documents.append({"content": content, "metadata": {"source": uploaded_file.name}})
            except Exception as e:
                st.sidebar.error(f"Error processing file {uploaded_file.name}: {e}")
        st.sidebar.success("Files processed successfully!")

elif input_type == "Images":
    # Image Upload
    st.sidebar.subheader("Upload Images")
    uploaded_images = st.sidebar.file_uploader(
        "Upload images with text (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_images:
        for image_file in uploaded_images:
            try:
                image = Image.open(image_file)
                content = pytesseract.image_to_string(image)
                documents.append({"content": content, "metadata": {"source": image_file.name}})
            except Exception as e:
                st.sidebar.error(f"Error processing image {image_file.name}: {e}")
        st.sidebar.success("Images processed successfully!")

# Process documents into vectorstore if any documents are present
if documents:
    try:
        # Split documents into smaller chunks
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(documents)

        # Create a vectorstore for retrieval
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name='rag-chroma',
            embedding=OllamaEmbeddings(model='nomic-embed-text')
        )
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.sidebar.error(f"Error creating vectorstore: {e}")

# Initialize session state for conversation history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat interface
st.subheader("Chat with the Deepseek-R1 Bot")
user_input = st.chat_input("Ask a question:")
if user_input:
    with st.spinner("Thinking..."):
        # Include chat history as context
        chat_history_context = "\n".join(
            [f"User: {chat['user']}\nAssistant: {chat['assistant']}" for chat in st.session_state["chat_history"]]
        )

        # Append the current question to the context
        full_context = f"{chat_history_context}\nUser: {user_input}\nAssistant:"

        response = None
        if retriever:
            # RAG-based response using context
            after_rag_template = """Answer the question based only on the following context:
            {context}
            Question: {question}
            """
            after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
            after_rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | after_rag_prompt
                | model_local
                | StrOutputParser()
            )
            response = after_rag_chain.invoke(user_input)
        else:
            # Simple response from the local model
            response = model_local.invoke(full_context)

        # Extract and clean the content
        try:
            response_content = response.content.strip()  # Extract the 'content' field and remove extra spaces
            response_content = response_content.replace("<think>", "").replace("</think>", "").strip()  # Remove tags
        except AttributeError:
            response_content = "I couldn't generate a proper response. Please try again."

        # Append the interaction to the chat history
        st.session_state["chat_history"].append({"user": user_input, "assistant": response_content})

# Display conversation history
for chat in st.session_state["chat_history"]:
    st.chat_message("user").write(chat["user"])
    st.chat_message("assistant").write(chat["assistant"])
