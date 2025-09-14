# Erste Schritte mit LangChain

## 🚀 LangChain Grundlagen verstehen

### Was ist LangChain?
LangChain ist ein Framework für die Entwicklung von Anwendungen mit Large Language Models (LLMs). Es bietet:
- **Chains**: Verkettung von LLM-Aufrufen
- **Agents**: Intelligente Entscheidungsfindung
- **Memory**: Konversationsgedächtnis
- **Tools**: Integration externer Services

### 1. Dein erstes LLM
```python
# first_llm.py
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

# LLM initialisieren
llm = OpenAI(temperature=0.7)

# Einfacher Aufruf
response = llm("Erkläre mir LangChain in zwei Sätzen")
print(response)
```

### 2. Chat Models vs LLMs
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Standard LLM - Text zu Text
llm = OpenAI()
response = llm("Schreibe ein Gedicht über Python")

# Chat Model - strukturierte Konversation
chat = ChatOpenAI()
messages = [
    SystemMessage(content="Du bist ein hilfreicher Python-Tutor"),
    HumanMessage(content="Erkläre mir list comprehensions")
]
response = chat(messages)
print(response.content)
```

### 3. Prompt Templates - der professionelle Weg
```python
from langchain import PromptTemplate
from langchain.llms import OpenAI

# Einfaches Template
template = """
Du bist ein {role}.
Beantworte folgende Frage: {question}
Antworte in {language}.
"""

prompt = PromptTemplate(
    input_variables=["role", "question", "language"],
    template=template
)

# Template verwenden
llm = OpenAI()
formatted_prompt = prompt.format(
    role="Python-Experte",
    question="Was sind die Vorteile von FastAPI?",
    language="Deutsch"
)

response = llm(formatted_prompt)
print(response)
```

### 4. Chain - dein erstes verkettetes System
```python
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Template definieren
prompt = PromptTemplate(
    input_variables=["product"],
    template="Schreibe einen kreativen Werbetext für: {product}"
)

# Chain erstellen
llm = OpenAI(temperature=0.9)
chain = LLMChain(llm=llm, prompt=prompt)

# Chain ausführen
result = chain.run("nachhaltige Sportschuhe")
print(result)
```

### 5. Sequential Chain - mehrere Schritte
```python
from langchain import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)

# Schritt 1: Idee generieren
first_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generiere eine kreative App-Idee zum Thema: {topic}"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# Schritt 2: Features beschreiben  
second_prompt = PromptTemplate(
    input_variables=["app_idea"],
    template="Liste die 5 wichtigsten Features dieser App auf: {app_idea}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# Chains verbinden
overall_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

# Ausführen
result = overall_chain.run("Umweltschutz")
print(result)
```

## 🛠️ Praktische Übungen

### Übung 1: Persönlicher Assistent
Erstelle einen einfachen Assistenten, der verschiedene Rollen übernehmen kann:

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

class PersonalAssistant:
    def __init__(self):
        self.llm = OpenAI(temperature=0.7)
        
        # Template für verschiedene Rollen
        self.template = PromptTemplate(
            input_variables=["role", "task", "context"],
            template="""
            Du bist ein {role}.
            Aufgabe: {task}
            Kontext: {context}
            
            Antwort:
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.template)
    
    def ask(self, role: str, task: str, context: str = ""):
        return self.chain.run(role=role, task=task, context=context)

# Testen
assistant = PersonalAssistant()

# Als Programmierer
code_help = assistant.ask(
    role="Python-Entwickler",
    task="Erkläre den Unterschied zwischen List und Tuple",
    context="Für einen Anfänger"
)
print("Code Help:", code_help)

# Als Übersetzer
translation = assistant.ask(
    role="Übersetzer",
    task="Übersetze 'Hello World' ins Deutsche und Französische"
)
print("Translation:", translation)
```

### Übung 2: Content Pipeline
Erstelle eine Pipeline für Content-Erstellung:

```python
from langchain import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def create_content_pipeline():
    llm = OpenAI(temperature=0.8)
    
    # Schritt 1: Thema analysieren
    analyze_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Analysiere das Thema '{topic}' und identifiziere die 3 wichtigsten Aspekte."
    )
    analyze_chain = LLMChain(llm=llm, prompt=analyze_prompt)
    
    # Schritt 2: Content erstellen
    content_prompt = PromptTemplate(
        input_variables=["analysis"],
        template="Schreibe einen informativen Blogpost basierend auf: {analysis}"
    )
    content_chain = LLMChain(llm=llm, prompt=content_prompt)
    
    # Schritt 3: SEO-Optimierung
    seo_prompt = PromptTemplate(
        input_variables=["content"],
        template="Erstelle 5 SEO-Keywords und einen Meta-Description für: {content}"
    )
    seo_chain = LLMChain(llm=llm, prompt=seo_prompt)
    
    # Pipeline zusammenbauen
    pipeline = SimpleSequentialChain(
        chains=[analyze_chain, content_chain, seo_chain],
        verbose=True
    )
    
    return pipeline

# Verwenden
pipeline = create_content_pipeline()
result = pipeline.run("Nachhaltige Software-Entwicklung")
```

## 🔍 Debug und Monitoring

### Verbose Mode aktivieren
```python
# Zeigt jeden Schritt der Chain
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
```

### Callbacks für Monitoring
```python
from langchain.callbacks import StdOutCallbackHandler

# Callback für detailliertes Logging
callback = StdOutCallbackHandler()
chain = LLMChain(
    llm=llm, 
    prompt=prompt,
    callbacks=[callback]
)
```

## ✅ Erste Schritte Checklist
- [ ] Erstes LLM erfolgreich verwendet
- [ ] Unterschied zwischen LLM und ChatModel verstanden
- [ ] Prompt Templates erstellt und verwendet
- [ ] Erste Chain gebaut und getestet
- [ ] Sequential Chain für mehrstufige Verarbeitung
- [ ] Praktische Übungen abgeschlossen

## 🎯 Nächste Schritte
Du bist jetzt bereit für die fortgeschrittenen Konzepte! 

**Nächstes Modul:** `04-llms-chatmodels` - Tiefer in die LLM-Welt eintauchen
