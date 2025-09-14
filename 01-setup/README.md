# Setup und Installation

## 🛠️ Umgebung vorbereiten

### 1. Virtual Environment erstellen
```bash
python -m venv langchain-env
cd langchain-env
Scripts\activate  # Windows
# source bin/activate  # Linux/Mac
```

### 2. LangChain installieren
```bash
pip install langchain langchain-openai langchain-anthropic
pip install jupyter notebook  # für interaktive Entwicklung
pip install python-dotenv     # für Umgebungsvariablen
pip install streamlit         # für Web-Apps später
```

### 3. API Keys einrichten
Erstelle eine `.env` Datei:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
```

### 4. Test-Installation
```python
# test_installation.py
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Test OpenAI (wenn API-Key verfügbar)
if os.getenv("OPENAI_API_KEY"):
    llm = OpenAI(temperature=0.9)
    response = llm("Sage 'Hallo' auf Deutsch")
    print(f"OpenAI Response: {response}")
else:
    print("OpenAI API Key nicht gefunden - das ist OK für den Anfang!")

print("LangChain ist erfolgreich installiert! 🎉")
```

## ⚡ Alternativen ohne API-Keys

### Lokale Modelle (Ollama)
```bash
# Ollama installieren (https://ollama.ai)
ollama pull llama2
```

```python
from langchain.llms import Ollama
llm = Ollama(model="llama2")
```

### Hugging Face Modelle (kostenlos)
```python
from langchain import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    model_kwargs={"temperature": 0.7}
)
```

## 🔍 Projekt-Struktur
```
langchain-projekt/
├── .env
├── requirements.txt
├── notebooks/
├── src/
└── data/
```

## ✅ Setup-Checklist
- [ ] Virtual Environment aktiviert
- [ ] LangChain installiert
- [ ] Jupyter Notebook funktioniert
- [ ] Mindestens ein LLM verfügbar (API oder lokal)
- [ ] Test-Script erfolgreich ausgeführt

**Nächster Schritt:** `02-python-basics` für relevante Python-Konzepte
