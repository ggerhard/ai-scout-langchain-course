# Python Basics für LangChain

## 🐍 Wichtige Python-Konzepte für LangChain

### 1. Async/Await (sehr wichtig!)
LangChain nutzt viel asynchrone Programmierung für bessere Performance:

```python
import asyncio
from langchain.llms import OpenAI

# Synchron
llm = OpenAI()
response = llm("Hallo Welt")

# Asynchron (schneller bei mehreren Anfragen)
async def async_example():
    llm = OpenAI()
    response = await llm.agenerate(["Frage 1", "Frage 2", "Frage 3"])
    return response

# Ausführen
asyncio.run(async_example())
```

### 2. Type Hints (LangChain nutzt diese intensiv)
```python
from typing import List, Dict, Optional
from langchain.schema import BaseMessage

def process_messages(messages: List[BaseMessage]) -> Dict[str, str]:
    """Verarbeitet eine Liste von Messages"""
    result: Dict[str, str] = {}
    for msg in messages:
        result[msg.type] = msg.content
    return result
```

### 3. Dataclasses und Pydantic Models
LangChain basiert stark auf Pydantic für Datenvalidierung:

```python
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100
    
    class Config:
        # Erlaubt zusätzliche Felder
        extra = "allow"

# Verwendung
request = ChatRequest(
    message="Erkläre Quantenphysik",
    temperature=0.9
)
```

### 4. Context Managers (für Ressourcen-Management)
```python
from contextlib import contextmanager

@contextmanager
def llm_session():
    """Context Manager für LLM Sessions"""
    print("Session gestartet")
    try:
        yield "session_object"
    finally:
        print("Session beendet")

# Verwendung
with llm_session() as session:
    # Arbeite mit LLM
    pass
```

### 5. Decorators (für Caching und Monitoring)
```python
from functools import wraps
import time

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

@timing_decorator
def slow_llm_call():
    # Simuliere langsamen LLM Call
    time.sleep(1)
    return "Response"
```

### 6. List/Dict Comprehensions (häufig in LangChain)
```python
# Messages verarbeiten
messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hallo"}]

# List comprehension
user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]

# Dict comprehension
message_dict = {i: msg["content"] for i, msg in enumerate(messages)}
```

### 7. Exception Handling für API Calls
```python
from langchain.llms import OpenAI
import openai

def safe_llm_call(prompt: str) -> str:
    try:
        llm = OpenAI(temperature=0.7)
        response = llm(prompt)
        return response
    except openai.error.RateLimitError:
        return "Rate limit erreicht - bitte warten"
    except openai.error.InvalidRequestError as e:
        return f"Ungültige Anfrage: {e}"
    except Exception as e:
        return f"Unerwarteter Fehler: {e}"
```

### 8. Environment Variables (sicher API Keys verwenden)
```python
import os
from dotenv import load_dotenv

load_dotenv()

# Sichere API Key Verwendung
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY nicht gefunden in .env")

# Mit Default-Werten
TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
```

## 🧪 Praktische Übung

Erstelle eine einfache Klasse für LLM-Interaktionen:

```python
import os
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from langchain.llms import OpenAI

@dataclass
class LLMConfig:
    temperature: float = 0.7
    max_tokens: int = 100
    model_name: str = "text-davinci-003"

class SimpleLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = OpenAI(
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    def generate(self, prompt: str) -> str:
        """Synchrone Generierung"""
        try:
            return self.llm(prompt)
        except Exception as e:
            return f"Fehler: {e}"
    
    async def agenerate(self, prompts: List[str]) -> List[str]:
        """Asynchrone Generierung für mehrere Prompts"""
        tasks = [self.llm.agenerate([prompt]) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return [result.generations[0][0].text for result in results]

# Verwendung
config = LLMConfig(temperature=0.9, max_tokens=50)
simple_llm = SimpleLLM(config)

response = simple_llm.generate("Erkläre Machine Learning in einem Satz")
print(response)
```

## 🎯 Was du jetzt können solltest:
- [ ] Async/Await verstehen und anwenden
- [ ] Type Hints für besseren Code nutzen
- [ ] Pydantic Models für Datenvalidierung
- [ ] Exception Handling für API Calls
- [ ] Environment Variables sicher verwenden

**Nächster Schritt:** `03-first-steps` - Deine ersten LangChain Schritte!
