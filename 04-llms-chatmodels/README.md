# LLMs und Chat Models verstehen

## 🧠 Was sind LLMs?

Large Language Models (LLMs) sind die Grundlage von LangChain. Sie verstehen und generieren menschliche Sprache. LangChain unterscheidet zwischen zwei Haupttypen:

- **LLMs**: Text-zu-Text (GPT-3, Claude)
- **Chat Models**: Nachrichtenbasiert (ChatGPT, Claude Chat)

## 1. LLM Grundlagen

### Verschiedene LLM Provider
```python
# OpenAI
from langchain.llms import OpenAI
openai_llm = OpenAI(
    temperature=0.7,
    max_tokens=100,
    model_name="gpt-3.5-turbo-instruct"
)

# Anthropic Claude
from langchain.llms import Anthropic
claude_llm = Anthropic(
    temperature=0.5,
    max_tokens=150
)

# HuggingFace (kostenlos)
from langchain.llms import HuggingFacePipeline
hf_llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation"
)

# Ollama (lokal)
from langchain.llms import Ollama
ollama_llm = Ollama(model="llama2")

# Test
response = openai_llm("Erkläre Machine Learning in 2 Sätzen:")
print(response)
```

### LLM Parameter verstehen
```python
# Temperature: Kreativität (0 = deterministisch, 1 = sehr kreativ)
creative_llm = OpenAI(temperature=0.9)
factual_llm = OpenAI(temperature=0.1)

print("Kreativ:", creative_llm("Schreibe eine Geschichte über Roboter"))
print("Faktisch:", factual_llm("Was ist die Hauptstadt von Deutschland?"))

# Max Tokens: Ausgabelänge begrenzen
short_llm = OpenAI(max_tokens=50)
long_llm = OpenAI(max_tokens=500)

# Top P: Alternative zu Temperature
top_p_llm = OpenAI(top_p=0.8, temperature=1)
```

## 2. Chat Models - Strukturierte Konversationen

### Basis Chat Model
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Chat Model initialisieren
chat = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# Einzelne Nachricht
response = chat([HumanMessage(content="Hallo, wie geht es dir?")])
print(response.content)

# Konversation mit Kontext
messages = [
    SystemMessage(content="Du bist ein hilfreicher Python-Tutor."),
    HumanMessage(content="Erkläre mir Listen in Python"),
    AIMessage(content="Listen sind geordnete, veränderbare Sammlungen..."),
    HumanMessage(content="Wie füge ich Elemente hinzu?")
]

response = chat(messages)
print(response.content)
```

### Verschiedene Message-Typen
```python
from langchain.schema import (
    SystemMessage,    # System-Instruktionen
    HumanMessage,     # Benutzer-Nachrichten  
    AIMessage,        # KI-Antworten
    FunctionMessage   # Tool/Function Ergebnisse
)

# System Message - Rolle definieren
system_msg = SystemMessage(
    content="Du bist ein Experte für deutsche Grammatik. Antworte präzise und hilfreich."
)

# Human Message - Benutzereingabe
human_msg = HumanMessage(
    content="Wann verwendet man 'das' und wann 'dass'?"
)

# AI Message - für Konversationshistorie
ai_msg = AIMessage(
    content="'Das' ist ein Artikel oder Pronomen, 'dass' ist eine Konjunktion..."
)

conversation = [system_msg, human_msg, ai_msg]
```

## 3. Streaming und Async

### Streaming für längere Antworten
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Streaming LLM
streaming_llm = OpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0.7
)

# Stream die Antwort
streaming_llm("Schreibe eine ausführliche Erklärung von Quantencomputern:")
```

### Asynchrone Verarbeitung
```python
import asyncio
from langchain.llms import OpenAI

async def async_llm_example():
    llm = OpenAI(temperature=0.5)
    
    # Mehrere Anfragen parallel
    questions = [
        "Was ist Python?",
        "Was ist JavaScript?", 
        "Was ist Go?"
    ]
    
    # Parallel verarbeiten
    tasks = [llm.agenerate([q]) for q in questions]
    results = await asyncio.gather(*tasks)
    
    for i, result in enumerate(results):
        print(f"Frage {i+1}: {questions[i]}")
        print(f"Antwort: {result.generations[0][0].text}\n")

# Ausführen
# asyncio.run(async_llm_example())
```

## 4. Custom LLM Wrapper

### Eigenen LLM Wrapper erstellen
```python
from langchain.llms.base import LLM
from typing import Optional, List
import requests

class CustomAPILLM(LLM):
    """Custom LLM für deine eigene API"""
    
    api_url: str
    api_key: str
    
    @property
    def _llm_type(self) -> str:
        return "custom_api"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Actual call to the API"""
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["text"]
        except Exception as e:
            return f"Error: {str(e)}"

# Verwendung
# custom_llm = CustomAPILLM(api_url="https://api.example.com/generate", api_key="your_key")
# result = custom_llm("Erkläre Blockchain")
```

### Offline LLM mit Transformers
```python
from transformers import pipeline
from langchain.llms.base import LLM

class LocalTransformerLLM(LLM):
    """Lokales LLM mit HuggingFace Transformers"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__()
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=-1  # CPU, nutze 0 für GPU
        )
    
    @property
    def _llm_type(self) -> str:
        return "local_transformer"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(
            prompt,
            max_length=len(prompt) + 100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        # Nur den generierten Teil zurückgeben
        generated_text = response[0]["generated_text"]
        return generated_text[len(prompt):].strip()

# Verwendung
# local_llm = LocalTransformerLLM()
# result = local_llm("KI ist")
```

## 5. Model Comparison und Benchmarking

### Verschiedene Modelle vergleichen
```python
from langchain.llms import OpenAI, Anthropic
from langchain.model_laboratory import ModelLaboratory

# Verschiedene LLMs
llms = [
    OpenAI(temperature=0.5, model_name="gpt-3.5-turbo-instruct"),
    OpenAI(temperature=0.5, model_name="text-davinci-003"),
    Anthropic(temperature=0.5)
]

# Model Laboratory
model_lab = ModelLaboratory.from_llms(llms)

# Vergleiche Antworten
model_lab.compare("Erkläre Quantenphysik in einfachen Worten:")
```

### Performance Benchmarking
```python
import time
from typing import List

def benchmark_llm(llm, questions: List[str]) -> dict:
    """Benchmarkt ein LLM"""
    
    results = {
        "model": llm._llm_type,
        "total_time": 0,
        "avg_time": 0,
        "responses": []
    }
    
    start_time = time.time()
    
    for question in questions:
        q_start = time.time()
        response = llm(question)
        q_end = time.time()
        
        results["responses"].append({
            "question": question,
            "response": response,
            "time": q_end - q_start,
            "length": len(response)
        })
    
    end_time = time.time()
    results["total_time"] = end_time - start_time
    results["avg_time"] = results["total_time"] / len(questions)
    
    return results

# Test Questions
test_questions = [
    "Was ist Machine Learning?",
    "Erkläre Python in 2 Sätzen",
    "Was sind die Vorteile von Cloud Computing?"
]

# Benchmark verschiedene LLMs
llm1 = OpenAI(temperature=0.5)
llm2 = Anthropic(temperature=0.5)

# results1 = benchmark_llm(llm1, test_questions)
# results2 = benchmark_llm(llm2, test_questions)

# print(f"OpenAI Average Time: {results1['avg_time']:.2f}s")
# print(f"Anthropic Average Time: {results2['avg_time']:.2f}s")
```

## 6. Error Handling und Retry Logic

### Robuste LLM Calls
```python
import time
from typing import Optional

class RobustLLM:
    def __init__(self, llm, max_retries: int = 3, delay: float = 1):
        self.llm = llm
        self.max_retries = max_retries
        self.delay = delay
    
    def call_with_retry(self, prompt: str) -> Optional[str]:
        """LLM Call mit Retry Logic"""
        
        for attempt in range(self.max_retries):
            try:
                result = self.llm(prompt)
                
                # Validierung der Antwort
                if len(result.strip()) < 5:
                    raise ValueError("Antwort zu kurz")
                
                return result
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"All {self.max_retries} attempts failed")
                    return f"Error: Could not generate response after {self.max_retries} attempts"
        
        return None

# Verwendung
robust_llm = RobustLLM(OpenAI(temperature=0.7))
result = robust_llm.call_with_retry("Erkläre Blockchain Technologie:")
print(result)
```

### Rate Limiting
```python
import time
from collections import deque

class RateLimitedLLM:
    def __init__(self, llm, max_calls_per_minute: int = 60):
        self.llm = llm
        self.max_calls_per_minute = max_calls_per_minute
        self.call_times = deque()
    
    def call(self, prompt: str) -> str:
        """Rate-limited LLM call"""
        
        now = time.time()
        
        # Entferne Calls älter als 1 Minute
        while self.call_times and self.call_times[0] < now - 60:
            self.call_times.popleft()
        
        # Prüfe Rate Limit
        if len(self.call_times) >= self.max_calls_per_minute:
            sleep_time = 60 - (now - self.call_times[0])
            print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        
        # Call ausführen
        self.call_times.append(now)
        return self.llm(prompt)

# Verwendung
rate_limited_llm = RateLimitedLLM(OpenAI(), max_calls_per_minute=10)
```

## 🎯 Praktische Übungen

### Übung 1: LLM Vergleich
```python
def compare_llm_creativity():
    """Vergleiche Kreativität verschiedener LLMs"""
    
    prompt = "Erfinde eine kurze Geschichte über einen Roboter, der Kaffee liebt:"
    
    llms = {
        "Conservative": OpenAI(temperature=0.1),
        "Balanced": OpenAI(temperature=0.7),
        "Creative": OpenAI(temperature=1.0)
    }
    
    for name, llm in llms.items():
        print(f"\n{name} (temp={llm.temperature}):")
        print("-" * 50)
        response = llm(prompt)
        print(response)

# compare_llm_creativity()
```

### Übung 2: Multi-Language Support
```python
def multilingual_llm():
    """LLM das mehrere Sprachen unterstützt"""
    
    chat = ChatOpenAI(temperature=0.5)
    
    languages = {
        "deutsch": "Erkläre Photosynthese",
        "english": "Explain photosynthesis", 
        "français": "Expliquez la photosynthèse",
        "español": "Explica la fotosíntesis"
    }
    
    for lang, question in languages.items():
        print(f"\n{lang.upper()}:")
        response = chat([HumanMessage(content=question)])
        print(response.content)

# multilingual_llm()
```

## ✅ LLM Mastery Checklist
- [ ] Verschiedene LLM Provider verstanden
- [ ] Parameter-Tuning (Temperature, Max Tokens)
- [ ] Chat Models vs Standard LLMs
- [ ] Streaming und Async implementiert
- [ ] Custom LLM Wrapper erstellt
- [ ] Model Benchmarking durchgeführt
- [ ] Error Handling implementiert
- [ ] Rate Limiting verstanden

## 🎯 Key Takeaways
- **Temperature**: 0 für Fakten, 1 für Kreativität
- **Chat Models**: Besser für Konversationen
- **Async**: Für bessere Performance bei mehreren Calls
- **Error Handling**: Essentiell für Produktionsumgebungen

**Nächstes Modul:** `05-prompts` - Professionelles Prompt Engineering
