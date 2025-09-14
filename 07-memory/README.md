# Memory - Konversationsgedächtnis

## 🧠 Was ist Memory in LangChain?

Memory ermöglicht es LangChain-Anwendungen, sich an vorherige Interaktionen zu erinnern. Ohne Memory behandelt jede Anfrage als komplett neue Konversation.

**Problem ohne Memory:** "Wie heißt du?" → "Ich bin Claude" → "Und wie alt bist du?" → "Wer bist du denn?"

**Lösung mit Memory:** Kontext bleibt erhalten!

## 1. ConversationBufferMemory - Die Basis

### Einfaches Gesprächsgedächtnis
```python
from langchain.memory import ConversationBufferMemory
from langchain import ConversationChain
from langchain.llms import OpenAI

# Memory initialisieren
memory = ConversationBufferMemory()

# LLM und Chain
llm = OpenAI(temperature=0.7)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # Zeigt Memory-Updates
)

# Konversation führen
print("Gespräch 1:")
response1 = conversation.predict(input="Hi, ich bin Max und arbeite als Entwickler.")
print(f"AI: {response1}")

print("\nGespräch 2:")
response2 = conversation.predict(input="Was weißt du über meinen Beruf?")
print(f"AI: {response2}")

print("\nGespräch 3:")
response3 = conversation.predict(input="Und wie heiße ich nochmal?")
print(f"AI: {response3}")

# Memory Inhalt anzeigen
print("\nMemory Buffer:")
print(memory.buffer)
```

### Memory manuell verwalten
```python
# Eigene Nachrichten hinzufügen
memory.save_context(
    {"input": "Mein Lieblingsprogrammiersprache ist Python"},
    {"output": "Das ist eine ausgezeichnete Wahl! Python ist sehr vielseitig."}
)

# Memory laden
memory_variables = memory.load_memory_variables({})
print("Memory Inhalt:", memory_variables['history'])

# Memory löschen
memory.clear()
print("Memory nach Clear:", memory.buffer)
```

## 2. ConversationBufferWindowMemory - Begrenzte Historie

### Sliding Window Memory
```python
from langchain.memory import ConversationBufferWindowMemory

# Nur letzten 3 Nachrichten-Paare merken
window_memory = ConversationBufferWindowMemory(k=3)

conversation_window = ConversationChain(
    llm=llm,
    memory=window_memory,
    verbose=True
)

# Viele Nachrichten senden
messages = [
    "Ich heiße Anna",
    "Ich bin 25 Jahre alt", 
    "Ich wohne in Berlin",
    "Ich arbeite als Designerin",
    "Mein Hobby ist Malen",
    "Wie heiße ich nochmal?"  # Wird Anna vergessen sein?
]

for msg in messages:
    response = conversation_window.predict(input=msg)
    print(f"User: {msg}")
    print(f"AI: {response}\n")

# Memory Inhalt checken
print("Window Memory Buffer:")
print(window_memory.buffer)
```

## 3. ConversationSummaryMemory - Intelligente Zusammenfassung

### Memory mit Zusammenfassungen
```python
from langchain.memory import ConversationSummaryMemory

# Summary Memory - fasst alte Nachrichten zusammen
summary_memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True
)

# Lange Konversation simulieren
long_conversation_inputs = [
    "Hi, ich bin Tom und entwickle Mobile Apps für iOS und Android.",
    "Mein aktuelles Projekt ist eine Fitness-App mit KI-Personal-Trainer.",
    "Die App nutzt Computer Vision um Übungen zu analysieren.",
    "Wir haben bereits 10.000 Beta-Nutzer und sehr positives Feedback.",
    "Die größte Herausforderung ist die Akku-Optimierung bei Video-Analyse.",
    "Was denkst du über unser Projekt?"
]

conversation_summary = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)

for input_text in long_conversation_inputs:
    response = conversation_summary.predict(input=input_text)
    print(f"User: {input_text}")
    print(f"AI: {response}\n")

# Summary anzeigen
print("Conversation Summary:")
print(summary_memory.moving_summary_buffer)
```

## 4. Memory mit Chains kombinieren

### Memory in verschiedenen Chain-Typen
```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# Memory-aware Prompt Template
memory_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
Bisherige Unterhaltung:
{history}

Aktuelle Anfrage: {input}

Beziehe dich auf vorherige Gespräche und antworte konsistent:
"""
)

# Chain mit Memory
memory_chain = LLMChain(
    llm=llm,
    prompt=memory_prompt,
    memory=ConversationBufferMemory(),
    verbose=True
)

# Konversations-Verlauf
responses = []
inputs = [
    "Ich plane eine Webseite für mein Restaurant",
    "Das Restaurant spezialisiert sich auf italienische Küche",
    "Welche Features sollte die Website haben?",
    "Wie kann ich Online-Bestellungen integrieren?"
]

for user_input in inputs:
    response = memory_chain.run(user_input)
    responses.append(response)
    print(f"User: {user_input}")
    print(f"AI: {response}\n")
```

## 5. Persistent Memory - Speicherung zwischen Sessions

### Memory mit Datei-Persistenz
```python
import json
import os
from datetime import datetime

class PersistentMemory:
    """Memory die zwischen Sessions gespeichert wird"""
    
    def __init__(self, user_id: str, memory_file: str = "memory.json"):
        self.user_id = user_id
        self.memory_file = memory_file
        self.memory = ConversationBufferMemory()
        self.load_from_file()
    
    def load_from_file(self):
        """Lädt Memory aus Datei"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if self.user_id in data:
                    user_data = data[self.user_id]
                    for entry in user_data.get('history', []):
                        self.memory.save_context(
                            {"input": entry["input"]},
                            {"output": entry["output"]}
                        )
                    print(f"✅ Memory für {self.user_id} geladen: {len(user_data.get('history', []))} Einträge")
                        
            except Exception as e:
                print(f"❌ Fehler beim Laden der Memory: {e}")
    
    def save_to_file(self):
        """Speichert Memory in Datei"""
        data = {}
        
        # Existierende Daten laden
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = {}
        
        # Memory für aktuellen User speichern
        history = []
        if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
            for i in range(0, len(self.memory.chat_memory.messages), 2):
                if i + 1 < len(self.memory.chat_memory.messages):
                    human_msg = self.memory.chat_memory.messages[i]
                    ai_msg = self.memory.chat_memory.messages[i + 1]
                    history.append({
                        "input": human_msg.content,
                        "output": ai_msg.content,
                        "timestamp": datetime.now().isoformat()
                    })
        
        data[self.user_id] = {
            "history": history,
            "last_updated": datetime.now().isoformat()
        }
        
        # In Datei speichern
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"💾 Memory für {self.user_id} gespeichert")
        except Exception as e:
            print(f"❌ Fehler beim Speichern: {e}")
    
    def get_conversation_chain(self, llm):
        """Erstellt ConversationChain mit persistent Memory"""
        return ConversationChain(
            llm=llm,
            memory=self.memory,
            verbose=False
        )
    
    def add_conversation(self, input_text: str, output_text: str):
        """Fügt Konversation zur Memory hinzu"""
        self.memory.save_context({"input": input_text}, {"output": output_text})
        self.save_to_file()

# Verwendung
user_memory = PersistentMemory("user_123")
conversation = user_memory.get_conversation_chain(llm)

# Erste Session
print("=== Erste Session ===")
response1 = conversation.predict(input="Ich bin Maria und arbeite als Data Scientist")
print(f"AI: {response1}")

response2 = conversation.predict(input="Ich arbeite hauptsächlich mit Python und R")
print(f"AI: {response2}")

# Simuliere Session-Ende
print("\n=== Session beendet - Memory gespeichert ===")

# Neue Session (Memory wird automatisch geladen)
print("\n=== Neue Session ===")
new_user_memory = PersistentMemory("user_123")
new_conversation = new_user_memory.get_conversation_chain(llm)

response3 = new_conversation.predict(input="Welche Programmiersprachen verwende ich?")
print(f"AI: {response3}")
```

## 6. Memory für RAG-Systeme

### RAG mit Konversational Memory
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

# Simuliere Dokumente (in Praxis: echte Dokumente laden)
documents = [
    "Python ist eine interpretierte Programmiersprache.",
    "JavaScript läuft im Browser und auf dem Server.",
    "Machine Learning ermöglicht Computern das Lernen aus Daten.",
    "APIs verbinden verschiedene Software-Systeme miteinander."
]

# Vector Store erstellen
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Simuliere Document Loading
from langchain.schema import Document
docs = [Document(page_content=doc) for doc in documents]
vectorstore = FAISS.from_documents(docs, embeddings)

# Memory für RAG
rag_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

# Conversational RAG Chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=rag_memory,
    return_source_documents=True,
    verbose=True
)

# RAG Konversation mit Memory
print("=== RAG mit Memory ===")
rag_questions = [
    "Was ist Python?",
    "Ist das eine interpretierte Sprache?",  # Bezieht sich auf vorherige Antwort
    "Welche anderen Programmiersprachen gibt es?",
    "Wie unterscheidet sich JavaScript von Python?"  # Vergleicht beide
]

for question in rag_questions:
    result = rag_chain({"question": question})
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print("---")
```

## 7. Memory Debugging und Monitoring

### Memory Inspector
```python
class MemoryInspector:
    """Tool zum Debuggen und Monitoren von Memory"""
    
    def __init__(self, memory):
        self.memory = memory
    
    def get_stats(self) -> dict:
        """Statistiken über Memory-Zustand"""
        if hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages
            return {
                "total_messages": len(messages),
                "human_messages": len([m for m in messages if hasattr(m, 'content') and 'Human' in str(type(m))]),
                "ai_messages": len([m for m in messages if hasattr(m, 'content') and 'AI' in str(type(m))]),
                "total_chars": sum(len(m.content) for m in messages if hasattr(m, 'content')),
                "memory_type": type(self.memory).__name__
            }
        elif hasattr(self.memory, 'buffer'):
            return {
                "buffer_length": len(self.memory.buffer),
                "memory_type": type(self.memory).__name__
            }
        return {"memory_type": type(self.memory).__name__}
    
    def print_memory_content(self):
        """Druckt Memory-Inhalt formatiert"""
        print(f"\n{'='*50}")
        print(f"MEMORY INSPECTOR - {type(self.memory).__name__}")
        print(f"{'='*50}")
        
        stats = self.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print(f"\n{'Memory Content:'}")
        print(f"{'-'*30}")
        
        if hasattr(self.memory, 'chat_memory'):
            for i, message in enumerate(self.memory.chat_memory.messages):
                msg_type = "👤 Human" if 'Human' in str(type(message)) else "🤖 AI"
                print(f"{i+1}. {msg_type}: {message.content[:100]}...")
        elif hasattr(self.memory, 'buffer'):
            print(self.memory.buffer)
        
        print(f"{'='*50}\n")
    
    def export_memory(self, filename: str):
        """Exportiert Memory in JSON-Datei"""
        export_data = {
            "memory_type": type(self.memory).__name__,
            "timestamp": datetime.now().isoformat(),
            "stats": self.get_stats(),
            "content": []
        }
        
        if hasattr(self.memory, 'chat_memory'):
            for message in self.memory.chat_memory.messages:
                export_data["content"].append({
                    "type": str(type(message).__name__),
                    "content": message.content
                })
        elif hasattr(self.memory, 'buffer'):
            export_data["content"] = self.memory.buffer
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Memory exportiert nach: {filename}")

# Verwendung des Inspectors
inspector = MemoryInspector(memory)
inspector.print_memory_content()
inspector.export_memory("memory_export.json")
```

## 🛠️ Praktisches Memory-Management System

```python
class AdvancedMemoryManager:
    """Fortgeschrittenes Memory-Management System"""
    
    def __init__(self, user_id: str, memory_type: str = "buffer"):
        self.user_id = user_id
        self.memory_type = memory_type
        self.memory = self._create_memory()
        self.inspector = MemoryInspector(self.memory)
        
    def _create_memory(self):
        """Erstellt Memory basierend auf Typ"""
        if self.memory_type == "buffer":
            return ConversationBufferMemory()
        elif self.memory_type == "window":
            return ConversationBufferWindowMemory(k=5)
        elif self.memory_type == "summary":
            return ConversationSummaryMemory(llm=OpenAI(temperature=0))
        else:
            return ConversationBufferMemory()
    
    def get_conversation_chain(self, llm):
        """Erstellt Conversation Chain"""
        return ConversationChain(
            llm=llm,
            memory=self.memory,
            verbose=False
        )
    
    def analyze_conversation(self, conversation_history: list):
        """Analysiert Konversations-Muster"""
        analysis = {
            "total_exchanges": len(conversation_history),
            "avg_input_length": 0,
            "avg_output_length": 0,
            "topics_mentioned": set(),
            "question_count": 0
        }
        
        input_lengths = []
        output_lengths = []
        
        for exchange in conversation_history:
            input_text = exchange.get('input', '')
            output_text = exchange.get('output', '')
            
            input_lengths.append(len(input_text))
            output_lengths.append(len(output_text))
            
            if '?' in input_text:
                analysis['question_count'] += 1
            
            # Simple Themen-Extraktion
            words = input_text.lower().split()
            topics = ['python', 'javascript', 'ai', 'ml', 'data', 'web', 'app']
            for topic in topics:
                if topic in words:
                    analysis['topics_mentioned'].add(topic)
        
        if input_lengths:
            analysis['avg_input_length'] = sum(input_lengths) / len(input_lengths)
            analysis['avg_output_length'] = sum(output_lengths) / len(output_lengths)
        
        analysis['topics_mentioned'] = list(analysis['topics_mentioned'])
        
        return analysis
    
    def optimize_memory(self):
        """Optimiert Memory basierend auf Usage Pattern"""
        stats = self.inspector.get_stats()
        
        recommendations = []
        
        if stats.get('total_messages', 0) > 50:
            recommendations.append("Erwäge Summary Memory für bessere Performance")
        
        if stats.get('total_chars', 0) > 10000:
            recommendations.append("Memory wird groß - Window Memory könnte effizienter sein")
        
        return recommendations

# Demo
memory_manager = AdvancedMemoryManager("user_456", "buffer")
llm = OpenAI(temperature=0.7)
conversation = memory_manager.get_conversation_chain(llm)

# Beispiel-Konversation
demo_inputs = [
    "Hi, ich lerne gerade Python programmieren",
    "Kannst du mir mit Listen helfen?",
    "Wie füge ich Elemente zu einer Liste hinzu?",
    "Was ist der Unterschied zwischen append() und extend()?"
]

print("=== Advanced Memory Management Demo ===")
for input_text in demo_inputs:
    response = conversation.predict(input=input_text)
    print(f"User: {input_text}")
    print(f"AI: {response}\n")

# Memory analysieren
memory_manager.inspector.print_memory_content()

# Optimierungsempfehlungen
recommendations = memory_manager.optimize_memory()
if recommendations:
    print("🔧 Optimierungsempfehlungen:")
    for rec in recommendations:
        print(f"- {rec}")
```

## ✅ Memory Mastery Checklist

- [ ] ConversationBufferMemory verstanden und implementiert
- [ ] Window Memory für begrenzte Historie
- [ ] Summary Memory für lange Konversationen
- [ ] Entity Memory für strukturierte Informationen
- [ ] Custom Memory-Implementierung erstellt
- [ ] Persistent Memory zwischen Sessions
- [ ] Memory mit RAG-Systemen kombiniert
- [ ] Memory Debugging und Monitoring implementiert

## 🎯 Best Practices

1. **Wähle den richtigen Memory-Typ**:
   - Buffer: Kurze Gespräche, vollständiger Kontext wichtig
   - Window: Mittlere Gespräche, nur neueste Nachrichten relevant
   - Summary: Lange Gespräche, Effizienz wichtig

2. **Performance beachten**: Große Memory kann LLM-Calls verlangsamen

3. **Privacy**: Sensible Daten in Memory berücksichtigen

4. **Persistenz**: Memory zwischen Sessions speichern für bessere UX

**Nächstes Modul:** `09-agents` - Intelligente Agenten entwickeln
