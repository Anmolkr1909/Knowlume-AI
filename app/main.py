
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import requests

def show_ollama_error(error):

    if isinstance(error, requests.exceptions.ConnectionError):

        st.error("Ollama is not running. Please open Ollama and try again.")

    elif isinstance(error, requests.exceptions.Timeout):

        st.error("The AI is taking too long. Please try again.")

    else:

        st.error("Ollama could not generate a response. Please try again.")

        st.caption(f"Technical details: {error}")

st.set_page_config(
    page_title="Knowlume AI",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    h1, h2, h3 {
        color: #172554;
    }

    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }

    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model

def generate_answer(question, context):

    prompt = f"""
    You are Knowlume AI, a document intelligence assistant.

    Answer the user's question using ONLY the provided document context.

    If the answer is not available in the context,
    clearly say that the document does not contain the answer.

    Document Context:
    {context}

    User Question:
    {question}

    Answer:
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"]

def generate_summary(document_text):

    prompt = f"""
    You are Knowlume AI, a document summarization assistant.

    Create a clear and useful summary of the following document.

    Include:
    1. Main topic
    2. Important points
    3. Key findings
    4. Important facts and figures
    5. Final conclusion

    Use simple language.
    Do not invent information.
    Only use the document content.

    DOCUMENT:
    {document_text}

    SUMMARY:
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"]


def compare_documents(document_a, document_b):

    prompt = f"""
    You are Knowlume AI, a document comparison assistant.

    Compare the following two documents.

    DOCUMENT A:
    {document_a}

    DOCUMENT B:
    {document_b}

    Explain:
    1. Common information
    2. Information only in Document A
    3. Information only in Document B
    4. Important differences

    Use simple and clear language.
    Do not invent information.
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"]


def clean_text(text):

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove repeated spaces
    while "  " in text:
        text = text.replace("  ", " ")

    # Remove unnecessary blank lines
    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    return text



def create_chunks(text, page_number, chunk_size=500, overlap=50):

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk_words = words[i:i + chunk_size]

        chunk_text = " ".join(chunk_words)

        chunk_data = {
            "chunk_id": len(chunks) + 1,
            "page": page_number,
            "text": chunk_text
        }

        chunks.append(chunk_data)

    return chunks

# Store uploaded documents between page navigation
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "document_pages" not in st.session_state:
    st.session_state.document_pages = []

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📚 Knowlume AI")
st.subheader("Illuminate the knowledge hidden in your documents.")

st.sidebar.title("📚 Knowlume AI")

st.sidebar.markdown(
    "**Document Intelligence Platform**"
)

st.sidebar.caption(
    "Illuminate the knowledge hidden in your documents."
)

st.sidebar.divider()

selected_page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "💬 Ask Documents",
        "📝 Document Summary",
        "📊 Compare Documents"
    ]
)


if selected_page == "🏠 Dashboard":

    st.title("🏠 Welcome to Knowlume AI")

    st.write(
        "Your intelligent workspace for understanding documents."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Platform", "AI Powered")

    with col2:
        st.metric("Features", "4")

    with col3:
        st.metric("Supported Format", "PDF")

    st.info(
        "Upload your documents and explore the power of AI."
    )

uploaded_files = st.session_state.uploaded_files

if selected_page == "🏠 Dashboard":

    st.header("Upload Your Document")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
st.session_state.uploaded_files = uploaded_files


if uploaded_files and selected_page == "🏠 Dashboard":

    st.success(f"{len(uploaded_files)} PDF file(s) uploaded successfully!")

    try:
        document_pages = []
        all_chunks = []
        chunk_embeddings = None

        for uploaded_file in uploaded_files:

            document_name = uploaded_file.name
            reader = PdfReader(uploaded_file)

            for page_number, page in enumerate(reader.pages, start=1):

                page_text = page.extract_text() or ""
                cleaned_page_text = clean_text(page_text)

                page_chunks = create_chunks(
                    cleaned_page_text,
                    page_number
                )

                for chunk in page_chunks:
                    chunk["document_name"] = document_name

                page_data = {
                    "document_name": document_name,
                    "page": page_number,
                    "text": cleaned_page_text,
                    "chunks": page_chunks
                }

                document_pages.append(page_data)
                all_chunks.extend(page_chunks)

        if all_chunks:

            embedding_model = load_embedding_model()

            chunk_texts = [
                chunk["text"] for chunk in all_chunks
            ]

            chunk_embeddings = embedding_model.encode(
                chunk_texts,
                convert_to_tensor=True
            )

        st.session_state.document_pages = document_pages
        st.session_state.all_chunks = all_chunks
        st.session_state.chunk_embeddings = chunk_embeddings
        
        document_pages = st.session_state.document_pages
        all_chunks = st.session_state.all_chunks
        chunk_embeddings = st.session_state.chunk_embeddings

        st.write("Total documents:", len(uploaded_files))
        st.write("Total pages:", len(document_pages))

        st.success("Document processing completed!")

    except Exception as e:
        st.error("Unable to process your PDF.")
        st.write(str(e))    


if selected_page == "📝 Document Summary":

    document_pages = st.session_state.document_pages
    uploaded_files = st.session_state.uploaded_files

    st.header("📝 Document Summary")

    if not uploaded_files or not document_pages:

        st.info("Please upload a PDF from the Dashboard first.")

    else:

        summary_document_name = st.selectbox(
            "Select a document to summarize",
            [file.name for file in uploaded_files],
            key="summary_document"
        )

        if st.button("Generate Summary"):

            summary_document = "\n\n".join(
                page["text"]
                for page in document_pages
                if page["document_name"] == summary_document_name
            )

            if not summary_document.strip():

                st.warning("No readable text found in this document.")

            else:

                with st.spinner("Generating document summary..."):

                    try:

                        summary = generate_summary(summary_document)

                        st.subheader("Document Summary")
                        st.write(summary)

                    except requests.exceptions.RequestException as e:

                        show_ollama_error(e)

if selected_page == "📊 Compare Documents":

    document_pages = st.session_state.document_pages
    uploaded_files = st.session_state.uploaded_files

    st.header("📊 Compare Documents")

    if len(uploaded_files) >= 2 and document_pages:

        document_names = [file.name for file in uploaded_files]

        col1, col2 = st.columns(2)

        with col1:
            document_a_name = st.selectbox(
                "Select First Document",
                document_names,
                key="document_a"
            )

        with col2:
            document_b_name = st.selectbox(
                "Select Second Document",
                document_names,
                index=1,
                key="document_b"
            )

        if st.button("Compare Documents"):

            if document_a_name == document_b_name:
                st.warning("Please select two different documents.")

            else:
                document_a = "\n\n".join(
                    page["text"] for page in document_pages
                    if page["document_name"] == document_a_name
                )

                document_b = "\n\n".join(
                    page["text"] for page in document_pages
                    if page["document_name"] == document_b_name
                )

                if not document_a.strip() or not document_b.strip():
                    st.warning("One or both documents contain no readable text.")

                else:
                    with st.spinner("Comparing documents..."):
                        try:
                            comparison = compare_documents(
                                document_a, document_b
                            )
                            st.subheader("Comparison Results")
                            st.write(comparison)

                        except requests.exceptions.RequestException as e:
                            st.error("Could not connect to Ollama.")
                            st.write(str(e))

    else:
        st.info("Upload at least 2 PDF documents from the Dashboard.")


if selected_page == "💬 Ask Documents":
        all_chunks = st.session_state.all_chunks
        chunk_embeddings = st.session_state.chunk_embeddings
        # Semantic Search + RAG
        st.header("Ask Knowlume AI")
        
        # Clear conversation history
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Display previous conversation
        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):
                st.write(message["content"])

                if message["role"] == "assistant" and "sources" in message:

                    st.caption("📚 Sources:")

                    for source in message["sources"]:
                        st.caption(f"• {source}")

        search_query = st.text_input(
            "Ask a question about your document"
        )

        if search_query and all_chunks and chunk_embeddings is not None:
            embedding_model = load_embedding_model()

            # Convert question into an embedding
            query_embedding = embedding_model.encode(
                search_query,
                convert_to_tensor=True
            )

            # Calculate similarity
            similarity_scores = util.cos_sim(
                query_embedding,
                chunk_embeddings
            )[0]

            # Retrieve top 3 relevant chunks
            top_results = similarity_scores.topk(
                k=min(3, len(all_chunks))
            )

            retrieved_chunks = []

            for score, index in zip(
                top_results.values,
                top_results.indices
            ):

                chunk = all_chunks[index.item()]
                retrieved_chunks.append(chunk)

                st.write("Source Page:", chunk["page"])
                st.write("Source Document:", chunk["document_name"])
                st.write("Similarity:", f"{score.item():.2f}")
                st.write(chunk["text"])
                st.divider()

            # Prepare context for the AI model
            context = "\n\n".join(
                [
                    f"Source: {chunk['document_name']} - Page {chunk['page']}\n{chunk['text']}"
                    for chunk in retrieved_chunks
                ]
            )

            # Generate answer using Qwen
            if st.button("Generate AI Answer"):

                with st.spinner("Knowlume AI is thinking..."):

                    try:
                        answer = generate_answer(
                            search_query,
                            context
                        )

                        st.subheader("Knowlume AI Answer")
                        st.write(answer)

                        # Save question and answer in chat history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": search_query
                        })

                        source_citations = sorted(
                            set(
                                f"{chunk['document_name']} — Page {chunk['page']}"
                                for chunk in retrieved_chunks
                            )
                        )

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": source_citations
                        })
                       

                        

                    except requests.exceptions.RequestException as e:
                        st.error("Could not connect to Ollama.")
                        
                        st.write(str(e))

        elif search_query:
            st.warning("No readable document chunks available.")



if selected_page == "🏠 Dashboard":
        
        document_pages = st.session_state.document_pages    
        st.header("Extracted Document Content")

        if document_pages:
            document_pages = st.session_state.document_pages

            for page_data in document_pages:

                with st.expander(
                    f"{page_data['document_name']} — Page {page_data['page']}"
                ):

                    if page_data["text"]:
                        st.write(page_data["text"])

                        st.write("Total chunks:", len(page_data["chunks"]))

                        for chunk in page_data["chunks"]:

                            with st.expander(f"Chunk {chunk['chunk_id']}"):

                                st.write("Chunk ID:", chunk["chunk_id"])

                                st.write("Page Number:", chunk["page"])

                                st.write(chunk["text"])

                    else:
                        st.warning("No readable text on this page.")

            st.success("Document processed successfully!")

        else:
            st.warning("This PDF has no pages.")

