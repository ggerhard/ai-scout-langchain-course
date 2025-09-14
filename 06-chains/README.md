# Chains - Das Herzstück von LangChain

## 🔗 Was sind Chains?

Chains sind das Kernkonzept von LangChain - sie ermöglichen es, mehrere LLM-Aufrufe und andere Operationen zu verketten. Statt einzelne, isolierte Anfragen zu stellen, können wir komplexe Workflows erstellen.

## 1. LLMChain - Die Basis

```python
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Einfache Chain
prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template="Schreibe eine Produktbeschreibung für {product} für {audience}"
)

llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)

# Verschiedene Verwendungsmethoden
result1 = chain.run(product="Smartwatch", audience="Sportler")
result2 = chain({"product": "E-Book", "audience": "Studenten"})
result3 = chain.apply([
    {"product": "Kaffee", "audience": "Büroarbeiter"},
    {"product": "Yoga-Matte", "audience": "Anfänger"}
])
```

## 2. Sequential Chains - Schritt für Schritt

### SimpleSequentialChain
```python
from langchain import SimpleSequentialChain

# Schritt 1: Story-Idee generieren
story_prompt = PromptTemplate(
    input_variables=["genre"],
    template="Erstelle eine kurze Story-Idee für das Genre: {genre}"
)
story_chain = LLMChain(llm=llm, prompt=story_prompt)

# Schritt 2: Charaktere entwickeln
character_prompt = PromptTemplate(
    input_variables=["story"],
    template="Erstelle 3 interessante Charaktere für diese Story: {story}"
)
character_chain = LLMChain(llm=llm, prompt=character_prompt)

# Chains verbinden
story_development = SimpleSequentialChain(
    chains=[story_chain, character_chain],
    verbose=True
)

result = story_development.run("Science Fiction")
```

## 3. Router Chain - Intelligente Weiterleitung

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain

# Verschiedene Expertenbereiche
physics_template = """Du bist ein Physikprofessor. 
Beantworte folgende Frage wissenschaftlich: {input}"""

cooking_template = """Du bist ein Spitzenkoch.
Gib praktische Kochratschläge für: {input}"""

programming_template = """Du bist ein Senior-Entwickler.
Erkläre das Programmierkonzept: {input}"""

# Router Chain setup
prompt_infos = [
    {"name": "physics", "description": "Physik und Naturwissenschaften", 
     "prompt_template": physics_template},
    {"name": "cooking", "description": "Kochen und Rezepte",
     "prompt_template": cooking_template},
    {"name": "programming", "description": "Programmierung und Software",
     "prompt_template": programming_template}
]
```

## 4. Custom Chain erstellen

```python
from langchain.chains.base import Chain
from typing import Dict, List
import re

class ContentOptimizationChain(Chain):
    """Custom Chain für Content-Optimierung"""
    
    llm_chain: LLMChain
    
    @property
    def input_keys(self) -> List[str]:
        return ["content", "target_audience"]
    
    @property  
    def output_keys(self) -> List[str]:
        return ["optimized_content", "seo_keywords", "readability_score"]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        content = inputs["content"]
        audience = inputs["target_audience"]
        
        # Schritt 1: Content optimieren
        optimize_prompt = f"""
        Optimiere folgenden Text für {audience}:
        {content}
        
        Mache ihn klarer, ansprechender und zielgruppengerecht.
        """
        
        optimized = self.llm_chain.run(optimize_prompt)
        
        # Schritt 2: SEO Keywords extrahieren  
        keyword_prompt = f"""
        Extrahiere 5 wichtige SEO-Keywords aus diesem Text:
        {optimized}
        
        Format: keyword1, keyword2, keyword3, keyword4, keyword5
        """
        
        keywords = self.llm_chain.run(keyword_prompt)
        
        # Schritt 3: Lesbarkeit bewerten
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_words = word_count / max(sentence_count, 1)
        
        if avg_words < 15:
            readability = "Sehr gut (einfach zu lesen)"
        elif avg_words < 20:
            readability = "Gut"
        else:
            readability = "Verbesserungsbedarf (zu komplexe Sätze)"
            
        return {
            "optimized_content": optimized,
            "seo_keywords": keywords,
            "readability_score": readability
        }

# Verwendung
llm = OpenAI(temperature=0.7)
base_chain = LLMChain(llm=llm, prompt=PromptTemplate(
    template="{input}", input_variables=["input"]
))

optimizer = ContentOptimizationChain(llm_chain=base_chain)

result = optimizer({
    "content": "Unsere Software ist sehr kompliziert aber gut.",
    "target_audience": "kleine Unternehmen"
})
```

## 5. Transformation Chain

```python
from langchain.chains import TransformChain

def transform_input(inputs: dict) -> dict:
    """Transformiert Input-Daten"""
    text = inputs["text"]
    
    # Text bereinigen
    cleaned_text = text.strip().replace("\n\n", "\n")
    
    # Wörter zählen
    word_count = len(cleaned_text.split())
    
    # Schwierigkeit einschätzen
    if word_count < 50:
        difficulty = "einfach"
    elif word_count < 200:
        difficulty = "mittel"
    else:
        difficulty = "schwer"
    
    return {
        "cleaned_text": cleaned_text,
        "word_count": word_count,
        "difficulty": difficulty
    }

# Transform Chain erstellen
transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["cleaned_text", "word_count", "difficulty"],
    transform=transform_input
)

# Mit LLM Chain kombinieren
analyze_prompt = PromptTemplate(
    template="""
    Analysiere diesen Text ({difficulty} - {word_count} Wörter):
    {cleaned_text}
    
    Gib eine strukturierte Analyse.
    """,
    input_variables=["cleaned_text", "word_count", "difficulty"]
)

analysis_chain = LLMChain(llm=llm, prompt=analyze_prompt)

# Chains verbinden
full_pipeline = SimpleSequentialChain(
    chains=[transform_chain, analysis_chain],
    verbose=True
)
```

## 6. Conditional Chain - If/Else Logik

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ConditionalSupportChain:
    def __init__(self, llm):
        self.llm = llm
        
        # Template für technische Fragen
        self.tech_template = PromptTemplate(
            input_variables=["question"],
            template="""
            Du bist ein technischer Support-Experte.
            Beantworte diese technische Frage Schritt für Schritt:
            {question}
            """
        )
        
        # Template für allgemeine Fragen
        self.general_template = PromptTemplate(
            input_variables=["question"],
            template="""
            Du bist ein freundlicher Kundenservice-Mitarbeiter.
            Beantworte diese Frage höflich und hilfsbereit:
            {question}
            """
        )
        
        self.tech_chain = LLMChain(llm=llm, prompt=self.tech_template)
        self.general_chain = LLMChain(llm=llm, prompt=self.general_template)
        
        # Keywords für technische Fragen
        self.tech_keywords = [
            "error", "fehler", "bug", "installation", "konfiguration",
            "api", "database", "server", "code", "programmierung"
        ]
    
    def classify_question(self, question: str) -> str:
        """Klassifiziert die Frage als technisch oder allgemein"""
        question_lower = question.lower()
        
        for keyword in self.tech_keywords:
            if keyword in question_lower:
                return "technical"
        
        return "general"
    
    def run(self, question: str) -> str:
        """Führt die entsprechende Chain basierend auf Klassifikation aus"""
        question_type = self.classify_question(question)
        
        if question_type == "technical":
            print("🔧 Technische Frage erkannt - Verwende Tech-Support Chain")
            return self.tech_chain.run(question)
        else:
            print("💬 Allgemeine Frage erkannt - Verwende General-Support Chain")
            return self.general_chain.run(question)

# Verwendung
support_chain = ConditionalSupportChain(llm)

# Tests
print(support_chain.run("Ich habe einen API-Fehler in meinem Code"))
print("---")
print(support_chain.run("Wann haben Sie geöffnet?"))
```

## 🛠️ Praktische Chain-Patterns

### 1. Retry Chain mit Fehlerbehandlung
```python
import time
from typing import Optional

class RetryChain:
    def __init__(self, chain: LLMChain, max_retries: int = 3):
        self.chain = chain
        self.max_retries = max_retries
    
    def run_with_retry(self, input_data: str) -> Optional[str]:
        """Führt Chain mit Retry-Logik aus"""
        
        for attempt in range(self.max_retries):
            try:
                result = self.chain.run(input_data)
                
                # Validierung des Results
                if len(result.strip()) < 10:
                    raise ValueError("Antwort zu kurz")
                
                return result
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return f"Fehler nach {self.max_retries} Versuchen: {e}"
        
        return None

# Verwendung
base_chain = LLMChain(llm=llm, prompt=PromptTemplate(
    template="Erkläre {topic} ausführlich",
    input_variables=["topic"]
))

retry_chain = RetryChain(base_chain, max_retries=3)
result = retry_chain.run_with_retry("Quantencomputer")
```

### 2. Caching Chain
```python
from functools import lru_cache
import hashlib

class CachingChain:
    def __init__(self, chain: LLMChain):
        self.chain = chain
        self.cache = {}
    
    def _get_cache_key(self, input_data: str) -> str:
        """Erstellt Cache-Schlüssel"""
        return hashlib.md5(input_data.encode()).hexdigest()
    
    def run(self, input_data: str) -> str:
        """Führt Chain mit Caching aus"""
        cache_key = self._get_cache_key(input_data)
        
        if cache_key in self.cache:
            print("🎯 Cache Hit!")
            return self.cache[cache_key]
        
        print("🔄 Cache Miss - Running chain...")
        result = self.chain.run(input_data)
        self.cache[cache_key] = result
        
        return result

# Verwendung
cached_chain = CachingChain(base_chain)
result1 = cached_chain.run("Erkläre Machine Learning")  # Cache Miss
result2 = cached_chain.run("Erkläre Machine Learning")  # Cache Hit
```

## ✅ Chains Mastery Checklist
- [ ] LLMChain verstanden und verwendet
- [ ] Sequential Chains für mehrstufige Verarbeitung
- [ ] Router Chains für intelligente Weiterleitung  
- [ ] Custom Chain erstellt
- [ ] Transform Chains für Datenverarbeitung
- [ ] Conditional Logic implementiert
- [ ] Retry und Caching Patterns angewendet

## 🎯 Praktische Aufgabe
Erstelle eine **Blog-Content-Pipeline** mit:
1. **Themen-Analyse** (Keywords extrahieren)
2. **Content-Erstellung** (Artikel schreiben)
3. **SEO-Optimierung** (Meta-Tags generieren)
4. **Social-Media-Posts** (Teaser erstellen)

**Nächstes Modul:** `07-memory` - Konversationsgedächtnis implementieren
