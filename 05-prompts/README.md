# Prompt Engineering mit LangChain

## 🎯 Was ist Prompt Engineering?

Prompt Engineering ist die Kunst, LLMs durch geschickte Anweisungen zu optimalen Ergebnissen zu führen. In LangChain geschieht das über **Prompt Templates** - wiederverwendbare, parametrisierte Prompts.

## 1. Prompt Template Grundlagen

### Einfache Templates
```python
from langchain.prompts import PromptTemplate

# Basis Template
simple_template = PromptTemplate(
    input_variables=["product"],
    template="Schreibe eine Produktbeschreibung für: {product}"
)

# Verwendung
prompt = simple_template.format(product="Smartwatch")
print(prompt)
# Output: "Schreibe eine Produktbeschreibung für: Smartwatch"
```

### Templates mit mehreren Variablen
```python
# Komplexeres Template
marketing_template = PromptTemplate(
    input_variables=["product", "audience", "tone"],
    template="""
Schreibe einen {tone} Werbetext für {product}.
Zielgruppe: {audience}

Der Text soll:
- Aufmerksamkeit erregen
- Die Hauptvorteile hervorheben
- Zum Kauf animieren

Werbetext:
"""
)

# Mit Chain verwenden
from langchain.llms import OpenAI
from langchain import LLMChain

llm = OpenAI(temperature=0.8)
marketing_chain = LLMChain(llm=llm, prompt=marketing_template)

result = marketing_chain.run(
    product="E-Bike",
    audience="Berufspendler",
    tone="professionell"
)
print(result)
```

## 2. Chat Prompt Templates

### Chat-spezifische Templates
```python
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# System Message Template
system_template = "Du bist ein {role}. Antworte {style}."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human Message Template  
human_template = "Bitte erkläre mir: {topic}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Chat Prompt kombinieren
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

# Verwenden
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.7)
formatted_prompt = chat_prompt.format_prompt(
    role="Physikprofessor",
    style="wissenschaftlich aber verständlich",
    topic="Quantenverschränkung"
)

response = chat(formatted_prompt.to_messages())
print(response.content)
```

## 3. Few-Shot Prompting

### Beispiel-basierte Templates
```python
from langchain.prompts.few_shot import FewShotPromptTemplate

# Beispiele definieren
examples = [
    {
        "input": "glücklich",
        "output": "Ich fühle mich heute richtig glücklich und voller Energie! 😊"
    },
    {
        "input": "müde", 
        "output": "Ich bin total müde und brauche dringend Schlaf... 😴"
    },
    {
        "input": "aufgeregt",
        "output": "Ich bin so aufgeregt! Das wird bestimmt fantastisch! 🎉"
    }
]

# Beispiel-Template
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Gefühl: {input}\nNachricht: {output}"
)

# Few-Shot Template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Schreibe emotionale Nachrichten basierend auf Gefühlen:",
    suffix="Gefühl: {input}\nNachricht:",
    input_variables=["input"],
    example_separator="\n\n"
)

# Verwenden
prompt = few_shot_prompt.format(input="nervös")
print(prompt)

# Mit LLM
llm = OpenAI(temperature=0.8)
response = llm(prompt)
print(f"Nervös: {response}")
```

### Dynamic Few-Shot mit Example Selector
```python
from langchain.prompts.example_selector import LengthBasedExampleSelector

# Viele Beispiele
examples = [
    {"input": "Python", "output": "Python ist eine vielseitige Programmiersprache"},
    {"input": "JavaScript", "output": "JavaScript läuft im Browser und auf Servern"},
    {"input": "Machine Learning", "output": "ML ermöglicht Computern das Lernen aus Daten"},
    {"input": "API", "output": "APIs verbinden verschiedene Software-Komponenten"},
    {"input": "Database", "output": "Datenbanken speichern strukturierte Informationen"}
]

# Example Selector - wählt basierend auf Länge
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_template,
    max_length=200  # Maximale Prompt-Länge
)

# Dynamic Few-Shot Template
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_template,
    prefix="Erkläre Tech-Begriffe kurz und prägnant:",
    suffix="Begriff: {input}\nErklärung:",
    input_variables=["input"]
)
```

## 4. Output Parser für strukturierte Antworten

### Pydantic Output Parser
```python
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

# Datenmodell definieren
class ProductReview(BaseModel):
    product_name: str = Field(description="Name des Produkts")
    rating: int = Field(description="Bewertung von 1-5 Sternen") 
    pros: List[str] = Field(description="Positive Aspekte")
    cons: List[str] = Field(description="Negative Aspekte")
    recommendation: bool = Field(description="Empfehlung ja/nein")

# Parser erstellen
parser = PydanticOutputParser(pydantic_object=ProductReview)

# Template mit Parser-Anweisungen
review_prompt = PromptTemplate(
    template="""
Analysiere folgende Produktbewertung und extrahiere strukturierte Informationen:

{review_text}

{format_instructions}
""",
    input_variables=["review_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Verwendung
review_text = """
Das iPhone 14 ist wirklich beeindruckend! Die Kamera macht fantastische Fotos 
und die Performance ist top. Leider ist der Akku nicht so langlebig wie erwartet 
und der Preis ist sehr hoch. Trotzdem würde ich es empfehlen.
"""

llm = OpenAI(temperature=0)
formatted_prompt = review_prompt.format(review_text=review_text)
response = llm(formatted_prompt)

# Parsen der Antwort
try:
    parsed_review = parser.parse(response)
    print(f"Produkt: {parsed_review.product_name}")
    print(f"Rating: {parsed_review.rating}/5")
    print(f"Pros: {', '.join(parsed_review.pros)}")
    print(f"Cons: {', '.join(parsed_review.cons)}")
    print(f"Empfohlen: {'Ja' if parsed_review.recommendation else 'Nein'}")
except Exception as e:
    print(f"Parsing Fehler: {e}")
```

### Einfache Output Parser
```python
from langchain.output_parsers import CommaSeparatedListOutputParser

# Lista Parser
list_parser = CommaSeparatedListOutputParser()

list_prompt = PromptTemplate(
    template="""
Liste 5 wichtige Vorteile von {technology}:

{format_instructions}
""",
    input_variables=["technology"],
    partial_variables={"format_instructions": list_parser.get_format_instructions()}
)

# Verwendung
prompt = list_prompt.format(technology="Cloud Computing")
response = llm(prompt)
parsed_list = list_parser.parse(response)

print("Vorteile:")
for i, benefit in enumerate(parsed_list, 1):
    print(f"{i}. {benefit.strip()}")
```

## 5. Prompt Engineering Best Practices

### Chain of Thought Prompting
```python
# Chain of Thought Template
cot_template = PromptTemplate(
    input_variables=["problem"],
    template="""
Löse das folgende Problem Schritt für Schritt:

Problem: {problem}

Schritt-für-Schritt Lösung:
1. Verstehe das Problem
2. Identifiziere die gegebenen Informationen
3. Bestimme was gesucht ist
4. Wähle die richtige Methode
5. Führe die Berechnung durch
6. Überprüfe das Ergebnis

Lösung:
"""
)

math_chain = LLMChain(llm=llm, prompt=cot_template)
result = math_chain.run("Ein Zug fährt 120 km in 2 Stunden. Wie schnell ist er?")
print(result)
```

### Role-Based Prompting
```python
def create_expert_prompt(role: str, expertise: str, task: str):
    """Erstellt rollenbasierte Prompts"""
    
    template = PromptTemplate(
        input_variables=["question"],
        template=f"""
Du bist ein erfahrener {role} mit {expertise} Jahren Erfahrung.
Deine Aufgabe: {task}

Antworte professionell und nutze deine Expertise.
Erkläre komplexe Konzepte verständlich.

Frage: {{question}}

Antwort als {role}:
"""
    )
    return template

# Verschiedene Experten
software_expert = create_expert_prompt(
    "Senior Software Entwickler", 
    "15", 
    "Code-Reviews und Architektur-Beratung"
)

marketing_expert = create_expert_prompt(
    "Marketing Manager",
    "10",
    "Strategieentwicklung und Kampagnen-Optimierung"
)

# Verwendung
question = "Wie optimiere ich die Performance meiner Web-App?"
expert_chain = LLMChain(llm=llm, prompt=software_expert)
result = expert_chain.run(question)
```

### Prompt Validation und Testing
```python
class PromptTester:
    def __init__(self, llm):
        self.llm = llm
        
    def test_prompt_variations(self, base_template: str, variations: dict, test_input: str):
        """Testet verschiedene Prompt-Varianten"""
        
        results = {}
        
        for name, template_str in variations.items():
            template = PromptTemplate(
                input_variables=["input"],
                template=template_str
            )
            
            chain = LLMChain(llm=self.llm, prompt=template)
            result = chain.run(test_input)
            
            results[name] = {
                "template": template_str,
                "result": result,
                "length": len(result),
                "quality_score": self._rate_quality(result)
            }
        
        return results
    
    def _rate_quality(self, response: str) -> int:
        """Einfache Qualitätsbewertung (1-5)"""
        # Basis-Metriken
        length_score = min(len(response) // 50, 3)  # 0-3 Punkte für Länge
        structure_score = 1 if any(word in response.lower() 
                                 for word in ['erstens', 'zweitens', 'zunächst']) else 0
        detail_score = 1 if len(response.split('.')) > 3 else 0
        
        return min(length_score + structure_score + detail_score, 5)

# Testen
tester = PromptTester(llm)

variations = {
    "basic": "Erkläre: {input}",
    "detailed": "Erkläre {input} ausführlich mit Beispielen:",
    "structured": "Erkläre {input}:\n1. Definition\n2. Hauptmerkmale\n3. Anwendungen\n4. Beispiele",
    "conversational": "Stell dir vor, ein Freund fragt dich: Was ist {input}? Antworte freundlich und verständlich:"
}

results = tester.test_prompt_variations(
    "Erkläre: {input}",
    variations,
    "Machine Learning"
)

# Beste Variante finden
best_prompt = max(results.items(), key=lambda x: x[1]['quality_score'])
print(f"Beste Prompt-Variante: {best_prompt[0]}")
print(f"Quality Score: {best_prompt[1]['quality_score']}")
```

## 6. Conditional Prompting

### Adaptive Prompts basierend auf Input
```python
from langchain.prompts import PromptTemplate

class AdaptivePrompt:
    def __init__(self, llm):
        self.llm = llm
        
        # Templates für verschiedene Komplexitätsstufen
        self.simple_template = PromptTemplate(
            input_variables=["topic"],
            template="Erkläre {topic} in einfachen Worten für Anfänger:"
        )
        
        self.advanced_template = PromptTemplate(
            input_variables=["topic"],
            template="Gib eine detaillierte technische Erklärung von {topic} mit Fachterminologie:"
        )
        
        self.conversational_template = PromptTemplate(
            input_variables=["topic"],
            template="Erkläre {topic} so, als würdest du mit einem Freund sprechen:"
        )
    
    def get_appropriate_template(self, user_input: str) -> PromptTemplate:
        """Wählt Template basierend auf User Input"""
        
        technical_indicators = [
            'algorithmus', 'implementation', 'architektur', 
            'optimierung', 'performance', 'skalierung'
        ]
        
        beginner_indicators = [
            'was ist', 'wie funktioniert', 'grundlagen', 
            'einfach erklärt', 'für anfänger'
        ]
        
        user_lower = user_input.lower()
        
        if any(indicator in user_lower for indicator in technical_indicators):
            return self.advanced_template
        elif any(indicator in user_lower for indicator in beginner_indicators):
            return self.simple_template
        else:
            return self.conversational_template
    
    def respond(self, user_input: str, topic: str) -> str:
        """Adaptive Antwort basierend auf User Input"""
        
        template = self.get_appropriate_template(user_input)
        chain = LLMChain(llm=self.llm, prompt=template)
        
        return chain.run(topic)

# Verwendung
adaptive = AdaptivePrompt(llm)

# Verschiedene Anfragen
print("ANFÄNGER:")
print(adaptive.respond("Was ist eigentlich Machine Learning?", "Machine Learning"))

print("\nTECHNISCH:")
print(adaptive.respond("Erkläre die Algorithmus-Optimierung bei Machine Learning", "Machine Learning"))

print("\nKONVERSATIONELL:")
print(adaptive.respond("Erzähl mir über Machine Learning", "Machine Learning"))
```

## 7. Prompt Chains für komplexe Workflows

### Multi-Step Prompt Chain
```python
class ContentCreationPipeline:
    def __init__(self, llm):
        self.llm = llm
        
        # Schritt 1: Themen-Analyse
        self.analysis_template = PromptTemplate(
            input_variables=["topic", "audience"],
            template="""
Analysiere das Thema '{topic}' für die Zielgruppe '{audience}'.

Analysiere:
1. Hauptinteressen der Zielgruppe
2. Wissenslevel (Anfänger/Fortgeschritten)  
3. Relevante Unterthemen
4. Potenzielle Fragen

Analyse:
"""
        )
        
        # Schritt 2: Content-Struktur
        self.structure_template = PromptTemplate(
            input_variables=["analysis"],
            template="""
Basierend auf dieser Analyse: {analysis}

Erstelle eine detaillierte Content-Struktur:
1. Einleitung (Hook)
2. Hauptpunkte (3-5)
3. Praktische Beispiele
4. Call-to-Action

Struktur:
"""
        )
        
        # Schritt 3: Content-Erstellung  
        self.content_template = PromptTemplate(
            input_variables=["structure"],
            template="""
Schreibe einen vollständigen Artikel basierend auf:
{structure}

Der Artikel soll:
- Engaging und informativ sein
- Klare Struktur haben
- Praktische Tipps enthalten
- Professionell aber zugänglich geschrieben sein

Artikel:
"""
        )
    
    def create_content(self, topic: str, audience: str) -> dict:
        """Vollständige Content-Erstellung Pipeline"""
        
        # Schritt 1: Analyse
        analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_template)
        analysis = analysis_chain.run(topic=topic, audience=audience)
        
        # Schritt 2: Struktur
        structure_chain = LLMChain(llm=self.llm, prompt=self.structure_template)
        structure = structure_chain.run(analysis=analysis)
        
        # Schritt 3: Content
        content_chain = LLMChain(llm=self.llm, prompt=self.content_template)
        content = content_chain.run(structure=structure)
        
        return {
            "topic": topic,
            "audience": audience,
            "analysis": analysis,
            "structure": structure,
            "content": content
        }

# Verwendung
pipeline = ContentCreationPipeline(llm)
result = pipeline.create_content("Künstliche Intelligenz", "kleine Unternehmen")

print("ANALYSE:")
print(result["analysis"])
print("\n" + "="*50 + "\n")
print("STRUKTUR:")  
print(result["structure"])
print("\n" + "="*50 + "\n")
print("CONTENT:")
print(result["content"])
```

## ✅ Prompt Engineering Mastery Checklist

- [ ] Basis Prompt Templates verstanden und angewendet
- [ ] Chat Prompt Templates für Konversationen
- [ ] Few-Shot Prompting implementiert  
- [ ] Output Parser für strukturierte Daten
- [ ] Chain of Thought Prompting angewendet
- [ ] Role-based Prompting implementiert
- [ ] Prompt Testing und Validation durchgeführt
- [ ] Conditional/Adaptive Prompting erstellt
- [ ] Multi-Step Prompt Chains gebaut

## 🎯 Best Practices Zusammenfassung

1. **Klarheit**: Sei spezifisch und eindeutig
2. **Kontext**: Gib relevante Hintergrundinformationen
3. **Struktur**: Nutze klare Formatierung und Gliederung
4. **Beispiele**: Few-Shot für konsistente Outputs
5. **Testing**: Teste verschiedene Varianten
6. **Iteration**: Verfeinere basierend auf Ergebnissen

**Nächstes Modul:** `07-memory` - Konversationsgedächtnis implementieren
