# RAG - Retrieval-Augmented Generation

## 🔍 Was ist RAG?

RAG (Retrieval-Augmented Generation) ermöglicht es LLMs, auf externe Wissensdatenbanken zuzugreifen. Statt nur auf Trainingsdaten angewiesen zu sein, kann das Modell relevante Dokumente finden und diese als Kontext für die Antwort nutzen.

**Workflow:** Frage → Relevante Dokumente finden → Kontext + Frage an LLM → Antwort

## 1. Dokumente laden und verarbeiten

### Document Loaders
```python
from langchain.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, 
    WebBaseLoader, DirectoryLoader
)

# Text-Datei laden
loader = TextLoader("dokument.txt", encoding="utf-8")
documents = loader.load()

# Ganzes Verzeichnis laden
dir_loader = DirectoryLoader(
    "./docs", 
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
all_docs = dir_loader.load()

# PDF laden
pdf_loader = PyPDFLoader("handbuch.pdf")
pdf_docs = pdf_loader.load()

print(f"Geladen: {len(all_docs)} Dokumente")
```

### Text Splitting - Dokumente aufteilen
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Text Splitter konfigurieren
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximale Chunk-Größe
    chunk_overlap=200,      # Überlappung zwischen Chunks
    separators=["\n\n", "\n", ". ", " ", ""]  # Trennzeichen
)

# Dokumente in Chunks aufteilen
chunks = text_splitter.split_documents(all_docs)
print(f"Erstellt: {len(chunks)} Text-Chunks")

# Chunk-Qualität prüfen
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}:")
    print(f"Größe: {len(chunk.page_content)} Zeichen")
    print(f"Inhalt: {chunk.page_content[:150]}...")
```

## 2. Embeddings und Vector Stores

### Embeddings erstellen
```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# HuggingFace Embeddings (kostenlos)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Für deutsche Texte optimiert
german_embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# Test Embedding
test_text = "Machine Learning ist ein Teilbereich der künstlichen Intelligenz"
embedding_vector = embeddings.embed_query(test_text)
print(f"Embedding-Dimension: {len(embedding_vector)}")
```

### Vector Stores einrichten
```python
from langchain.vectorstores import FAISS, Chroma

# FAISS (lokal, sehr schnell)
print("Erstelle FAISS Vector Store...")
faiss_db = FAISS.from_documents(chunks, embeddings)

# Speichern für später
faiss_db.save_local("faiss_index")

# Wieder laden
faiss_db = FAISS.load_local("faiss_index", embeddings)

# Chroma (persistent)
chroma_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector Store erstellt!")
```

## 3. RAG Chain implementieren

### Einfache RAG Chain
```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Retriever erstellen
retriever = faiss_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# RAG Chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True
)

# Frage stellen
query = "Was sind die Hauptvorteile von Machine Learning?"
result = qa_chain({"query": query})

print("Antwort:", result["result"])
print("\nQuellen:")
for i, doc in enumerate(result["source_documents"]):
    print(f"Quelle {i+1}: {doc.page_content[:200]}...")
```

### Custom RAG mit Prompt Template
```python
from langchain.prompts import PromptTemplate

# Custom Prompt für bessere Antworten
custom_prompt = PromptTemplate(
    template="""Du bist ein hilfreicher Assistent. Verwende den folgenden Kontext, um die Frage zu beantworten.
Wenn du die Antwort im Kontext nicht findest, sage es ehrlich.

Kontext:
{context}

Frage: {question}

Antwort auf Deutsch:""",
    input_variables=["context", "question"]
)

# RAG Chain mit custom prompt
qa_chain_custom = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)
```

### Conversational RAG (mit Memory)
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Memory für Konversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Conversational RAG Chain
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True
)

# Multi-Turn Conversation
questions = [
    "Was ist Machine Learning?",
    "Welche Arten gibt es davon?",
    "Wie unterscheidet sich das von Deep Learning?"
]

for question in questions:
    result = conv_chain({"question": question})
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
```

## 4. Erweiterte RAG-Techniken

### Multi-Query Retrieval
```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Erstellt automatisch mehrere Varianten der Frage
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=faiss_db.as_retriever(),
    llm=llm,
    verbose=True
)

# Test
docs = multi_query_retriever.get_relevant_documents(
    "Erkläre neuronale Netzwerke"
)
print(f"Gefunden: {len(docs)} relevante Dokumente")
```

### Contextual Compression
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Compressor - extrahiert nur relevante Teile
compressor = LLMChainExtractor.from_llm(llm)

# Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=faiss_db.as_retriever(search_kwargs={"k": 10})
)

# Komprimierte Dokumente abrufen
compressed_docs = compression_retriever.get_relevant_documents(
    "Was sind Convolutional Neural Networks?"
)

print(f"Komprimiert auf: {len(compressed_docs)} Dokumente")
```

## 🛠️ Praktisches RAG-System

```python
class DocumentQASystem:
    def __init__(self, docs_path: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )
        self.llm = OpenAI(temperature=0)
        
        # Dokumente laden
        loader = DirectoryLoader(docs_path, glob="**/*.txt")
        documents = loader.load()
        
        # Chunks erstellen
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.chunks = splitter.split_documents(documents)
        
        # Vector Store
        self.db = FAISS.from_documents(self.chunks, self.embeddings)
        
        # QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.db.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        
        print(f"✅ System initialisiert mit {len(self.chunks)} Chunks")
    
    def ask(self, question: str):
        """Stelle eine Frage an das System"""
        result = self.qa_chain({"query": question})
        
        print(f"🤔 Frage: {question}")
        print(f"💡 Antwort: {result['result']}")
        print("📚 Quellen:")
        
        for i, doc in enumerate(result['source_documents'][:2]):
            print(f"   {i+1}. {doc.page_content[:100]}...")
        
        return result

# Verwendung
# qa_system = DocumentQASystem("./docs")
# qa_system.ask("Was ist Machine Learning?")
```

## ✅ RAG Mastery Checklist
- [ ] Dokumente laden und verarbeiten
- [ ] Text-Splitting verstehen und optimieren
- [ ] Embeddings erstellen und vergleichen
- [ ] Vector Stores einrichten (FAISS, Chroma)
- [ ] Basis-RAG Chain implementiert
- [ ] Conversational RAG mit Memory
- [ ] Erweiterte Retrieval-Methoden getestet
- [ ] RAG für verschiedene Datenquellen
- [ ] Evaluation und Optimierung durchgeführt

## 🎯 Praktisches Projekt
Erstelle ein **Knowledge-Base-System** für dein Unternehmen:
- Lade Handbücher, Wikis, FAQs
- Implementiere intelligente Suche
- Baue Web-Interface mit Streamlit
- Tracke Nutzer-Feedback für Verbesserungen

**Nächstes Modul:** `09-agents` - Intelligente Agenten entwickeln
