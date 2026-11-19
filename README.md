# 📚 Knowlume AI

### Illuminate the knowledge hidden in your documents.

Knowlume AI is an AI-powered Document Intelligence Platform that helps users understand, search, summarize, and compare PDF documents.

## 🚀 Features

- PDF document upload and text extraction
- Text cleaning and intelligent chunking
- Semantic search using Sentence Transformers
- Retrieval-Augmented Generation (RAG)
- AI-powered question answering
- Source citations with PDF names and page numbers
- Document summarization
- Multiple document comparison
- Chat history
- Local AI inference using Ollama and Qwen 2.5

## 🛠️ Technology Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- PyTorch
- Ollama
- Qwen 2.5 (0.5B)

## 📂 Project Structure

```text
KnowlumeAI/
├── app/
│   └── main.py
├── data/
├── tests/
├── utils/
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Enter the project folder:

```bash
cd KnowlumeAI
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install Ollama and download the model:

```bash
ollama pull qwen2.5:0.5b
```

5. Start the application:

```bash
python -m streamlit run app/main.py
```

## 🧠 How It Works

1. User uploads PDF documents.
2. Knowlume AI extracts and cleans document text.
3. Text is divided into smaller chunks.
4. Sentence Transformers generates embeddings.
5. Semantic search retrieves relevant chunks.
6. Retrieved information is passed to the Qwen model.
7. The AI generates an answer using document context.
8. Source document names and page numbers are displayed.

## 🎯 Project Objective

To make document understanding faster, easier, and more accessible through AI-powered document intelligence.

## 👨‍💻 Developer

Developed as an AI/ML project using Python, RAG, NLP, and Large Language Models.
