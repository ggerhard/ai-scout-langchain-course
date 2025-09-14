# Agents - Intelligente Entscheidungsfindung

## 🤖 Was sind Agents?

Agents sind autonome LLM-basierte Systeme, die **Werkzeuge verwenden** und **Entscheidungen treffen** können. Während Chains einen festen Ablauf haben, können Agents dynamisch entscheiden, welche Tools sie verwenden und wie sie Probleme lösen.

**Chain:** Immer A → B → C
**Agent:** Analysiert Problem → Wählt Tools → Führt aus → Evaluiert → Wiederholt bei Bedarf

## 1. Agent Grundlagen

### ReAct Agent - Reasoning + Acting
```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool

# Einfache Tools definieren
def get_weather(location: str) -> str:
    """Gibt das Wetter für einen Ort zurück (Simulation)"""
    weather_data = {
        "Berlin": "Sonnig, 22°C",
        "Munich": "Bewölkt, 18°C", 
        "Hamburg": "Regnerisch, 15°C"
    }
    return weather_data.get(location, f"Wetter für {location} unbekannt")

def calculate(expression: str) -> str:
    """Führt mathematische Berechnungen durch"""
    try:
        result = eval(expression)  # In Produktion: sicherer Parser verwenden
        return f"Das Ergebnis von {expression} ist {result}"
    except:
        return "Fehler bei der Berechnung"

def search_web(query: str) -> str:
    """Simuliert Web-Suche"""
    return f"Web-Suchergebnisse für '{query}': Relevante Informationen gefunden..."

# Tools für Agent erstellen
tools = [
    Tool(
        name="Wetter",
        func=get_weather,
        description="Nützlich um das aktuelle Wetter für eine Stadt abzufragen"
    ),
    Tool(
        name="Rechner",
        func=calculate,
        description="Nützlich für mathematische Berechnungen und Formeln"
    ),
    Tool(
        name="WebSuche",
        func=search_web,
        description="Nützlich um aktuelle Informationen im Internet zu finden"
    )
]

# LLM und Agent initialisieren
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3
)

# Agent testen
print("=== Agent Test ===")
response = agent.run("Wie ist das Wetter in Berlin und was ist 15 * 23?")
print(f"Antwort: {response}")
```

## 2. Custom Tools erstellen

### Fortgeschrittene Tools
```python
from langchain.tools import BaseTool
from typing import Optional
import json
import requests

class DatabaseTool(BaseTool):
    """Custom Tool für Datenverwaltung"""
    
    name = "Datenbank"
    description = "Führt Datenbankoperationen aus: suchen, speichern, aktualisieren"
    
    def __init__(self):
        super().__init__()
        # Simuliere In-Memory Datenbank
        self.data = {
            "users": [
                {"id": 1, "name": "Anna", "role": "Developer"},
                {"id": 2, "name": "Max", "role": "Designer"},
                {"id": 3, "name": "Sarah", "role": "Manager"}
            ],
            "projects": [
                {"id": 1, "name": "Website Redesign", "status": "active"},
                {"id": 2, "name": "Mobile App", "status": "completed"}
            ]
        }
    
    def _run(self, query: str) -> str:
        """Führt Datenbank-Query aus"""
        try:
            # Einfaches Query-Parsing
            query_lower = query.lower()
            
            if "users" in query_lower:
                if "developer" in query_lower:
                    users = [u for u in self.data["users"] if u["role"] == "Developer"]
                    return f"Developer gefunden: {json.dumps(users, indent=2)}"
                else:
                    return f"Alle Users: {json.dumps(self.data['users'], indent=2)}"
            
            elif "projects" in query_lower:
                if "active" in query_lower:
                    projects = [p for p in self.data["projects"] if p["status"] == "active"]
                    return f"Aktive Projekte: {json.dumps(projects, indent=2)}"
                else:
                    return f"Alle Projekte: {json.dumps(self.data['projects'], indent=2)}"
            
            return "Unterstützte Queries: 'users', 'projects', 'active projects', 'developer users'"
            
        except Exception as e:
            return f"Datenbankfehler: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class TaskManagerTool(BaseTool):
    """Tool für Aufgabenverwaltung"""
    
    name = "Aufgaben"
    description = "Verwaltet Aufgaben: erstellen, anzeigen, erledigen"
    
    def __init__(self):
        super().__init__()
        self.tasks = []
        self.task_counter = 1
    
    def _run(self, command: str) -> str:
        """Führt Aufgaben-Kommandos aus"""
        parts = command.split(' ', 1)
        action = parts[0].lower()
        
        if action == "create" and len(parts) > 1:
            task_desc = parts[1]
            task = {
                "id": self.task_counter,
                "description": task_desc,
                "status": "open",
                "created": "heute"
            }
            self.tasks.append(task)
            self.task_counter += 1
            return f"✅ Aufgabe #{task['id']} erstellt: {task_desc}"
        
        elif action == "list":
            if not self.tasks:
                return "Keine Aufgaben vorhanden"
            
            task_list = []
            for task in self.tasks:
                status_icon = "✅" if task["status"] == "done" else "📝"
                task_list.append(f"{status_icon} #{task['id']}: {task['description']}")
            
            return "Aktuelle Aufgaben:\n" + "\n".join(task_list)
        
        elif action == "complete" and len(parts) > 1:
            try:
                task_id = int(parts[1])
                for task in self.tasks:
                    if task["id"] == task_id:
                        task["status"] = "done"
                        return f"✅ Aufgabe #{task_id} als erledigt markiert"
                return f"Aufgabe #{task_id} nicht gefunden"
            except ValueError:
                return "Ungültige Aufgaben-ID"
        
        return "Kommandos: create <beschreibung>, list, complete <id>"
    
    async def _arun(self, command: str) -> str:
        return self._run(command)

class CalculatorTool(BaseTool):
    """Erweiterte Rechner-Tool"""
    
    name = "Erweiteter_Rechner"
    description = "Führt komplexe mathematische Berechnungen durch, einschließlich Statistik"
    
    def _run(self, expression: str) -> str:
        """Führt erweiterte Berechnungen durch"""
        try:
            import math
            import statistics
            
            # Sicherheitsprüfung
            allowed_chars = set('0123456789+-*/.() ,[]')
            allowed_words = ['math', 'sin', 'cos', 'tan', 'log', 'sqrt', 'statistics', 'mean', 'median']
            
            expression_lower = expression.lower()
            
            if "mean" in expression_lower or "durchschnitt" in expression_lower:
                # Extrahiere Zahlen für Durchschnittsberechnung
                numbers = [float(x) for x in expression.split() if x.replace('.', '').replace(',', '').isdigit()]
                if numbers:
                    result = statistics.mean(numbers)
                    return f"Durchschnitt von {numbers}: {result}"
                else:
                    return "Keine gültigen Zahlen für Durchschnittsberechnung gefunden"
            
            elif "sqrt" in expression_lower or "wurzel" in expression_lower:
                # Quadratwurzel
                number = float(expression.split()[-1])
                result = math.sqrt(number)
                return f"Quadratwurzel von {number}: {result}"
            
            else:
                # Standard mathematische Auswertung
                # Ersetze math functions
                expression = expression.replace('sin', 'math.sin')
                expression = expression.replace('cos', 'math.cos')
                expression = expression.replace('tan', 'math.tan')
                expression = expression.replace('log', 'math.log')
                expression = expression.replace('sqrt', 'math.sqrt')
                
                result = eval(expression)
                return f"Ergebnis: {result}"
                
        except Exception as e:
            return f"Berechnungsfehler: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        return self._run(expression)

# Custom Tools kombinieren
advanced_tools = [
    DatabaseTool(),
    TaskManagerTool(),
    CalculatorTool()
]

# Agent mit erweiterten Tools
advanced_agent = initialize_agent(
    tools=advanced_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Test der erweiterten Tools
print("=== Advanced Agent Demo ===")
response1 = advanced_agent.run("Erstelle eine Aufgabe 'LangChain Tutorial abschließen'")
print(f"Response 1: {response1}")

response2 = advanced_agent.run("Zeige mir alle Developer in der Datenbank")
print(f"Response 2: {response2}")

response3 = advanced_agent.run("Berechne den Durchschnitt von 15, 23, 31, 18, 27")
print(f"Response 3: {response3}")
```

## 3. Agent Memory und State Management

### Stateful Agent mit Memory
```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor

class StatefulAgent:
    """Agent mit Memory und State Management"""
    
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.state = {
            "current_project": None,
            "user_preferences": {},
            "session_data": {}
        }
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
    
    def run_with_context(self, query: str) -> str:
        """Führt Query mit State-Kontext aus"""
        
        # State-Information zu Query hinzufügen
        context_info = []
        
        if self.state["current_project"]:
            context_info.append(f"Aktuelles Projekt: {self.state['current_project']}")
        
        if self.state["user_preferences"]:
            prefs = ", ".join([f"{k}: {v}" for k, v in self.state["user_preferences"].items()])
            context_info.append(f"User-Präferenzen: {prefs}")
        
        if context_info:
            enriched_query = f"Kontext: {'; '.join(context_info)}\n\nAnfrage: {query}"
        else:
            enriched_query = query
        
        # Agent ausführen
        response = self.agent.run(enriched_query)
        
        # State basierend auf Interaktion aktualisieren
        self._update_state(query, response)
        
        return response
    
    def _update_state(self, query: str, response: str):
        """Aktualisiert State basierend auf Interaktion"""
        query_lower = query.lower()
        
        # Projekt-Kontext extrahieren
        if "projekt" in query_lower:
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() == "projekt" and i < len(words) - 1:
                    self.state["current_project"] = words[i + 1]
        
        # User-Präferenzen extrahieren
        if "bevorzuge" in query_lower or "mag" in query_lower:
            if "python" in query_lower:
                self.state["user_preferences"]["language"] = "Python"
            elif "javascript" in query_lower:
                self.state["user_preferences"]["language"] = "JavaScript"
        
        # Session-Daten sammeln
        self.state["session_data"]["last_query"] = query
        self.state["session_data"]["interactions"] = self.state["session_data"].get("interactions", 0) + 1

# Stateful Agent verwenden
task_tool = TaskManagerTool()
db_tool = DatabaseTool()

stateful_agent = StatefulAgent([task_tool, db_tool], llm)

print("=== Stateful Agent Demo ===")
response1 = stateful_agent.run_with_context("Ich arbeite an Projekt WebApp und bevorzuge Python")
print(f"Response 1: {response1}")

response2 = stateful_agent.run_with_context("Erstelle eine Aufgabe für das aktuelle Projekt")
print(f"Response 2: {response2}")

response3 = stateful_agent.run_with_context("Welche Entwickler arbeiten in meiner bevorzugten Sprache?")
print(f"Response 3: {response3}")

print("\nAgent State:")
print(f"Current Project: {stateful_agent.state['current_project']}")
print(f"User Preferences: {stateful_agent.state['user_preferences']}")
print(f"Session Data: {stateful_agent.state['session_data']}")
```

## 4. Multi-Agent System

### Agents die zusammenarbeiten
```python
class SpecializedAgent:
    """Spezialisierter Agent für bestimmte Aufgaben"""
    
    def __init__(self, name: str, specialization: str, tools: list, llm):
        self.name = name
        self.specialization = specialization
        self.tools = tools
        self.llm = llm
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )
    
    def can_handle(self, query: str) -> bool:
        """Prüft ob Agent die Anfrage bearbeiten kann"""
        query_lower = query.lower()
        
        if self.specialization == "math":
            math_keywords = ["berechne", "rechne", "mathematik", "summe", "durchschnitt", "+", "*", "/", "-"]
            return any(keyword in query_lower for keyword in math_keywords)
        
        elif self.specialization == "data":
            data_keywords = ["daten", "database", "user", "projekt", "suche", "finde"]
            return any(keyword in query_lower for keyword in data_keywords)
        
        elif self.specialization == "tasks":
            task_keywords = ["aufgabe", "task", "todo", "erledige", "erstelle"]
            return any(keyword in query_lower for keyword in task_keywords)
        
        return False
    
    def handle_query(self, query: str) -> str:
        """Bearbeitet eine Anfrage"""
        try:
            return self.agent.run(f"Als {self.name} ({self.specialization} Spezialist): {query}")
        except Exception as e:
            return f"Fehler beim Bearbeiten der Anfrage: {str(e)}"

class AgentOrchestrator:
    """Koordiniert mehrere spezialisierte Agents"""
    
    def __init__(self, llm):
        self.llm = llm
        self.agents = []
        self.conversation_history = []
    
    def add_agent(self, agent: SpecializedAgent):
        """Fügt einen Agent hinzu"""
        self.agents.append(agent)
        print(f"✅ {agent.name} ({agent.specialization}) hinzugefügt")
    
    def route_query(self, query: str) -> str:
        """Leitet Anfrage an geeigneten Agent weiter"""
        
        # Finde geeignete Agents
        capable_agents = [agent for agent in self.agents if agent.can_handle(query)]
        
        if not capable_agents:
            return "❌ Kein Agent kann diese Anfrage bearbeiten"
        
        elif len(capable_agents) == 1:
            # Ein Agent gefunden
            selected_agent = capable_agents[0]
            print(f"🎯 Anfrage an {selected_agent.name} weitergeleitet")
            response = selected_agent.handle_query(query)
            
        else:
            # Mehrere Agents gefunden - wähle den besten
            print(f"🔄 Mehrere Agents verfügbar: {[a.name for a in capable_agents]}")
            selected_agent = capable_agents[0]  # Vereinfacht: nehme ersten
            print(f"🎯 {selected_agent.name} ausgewählt")
            response = selected_agent.handle_query(query)
        
        # History aktualisieren
        self.conversation_history.append({
            "query": query,
            "agent": selected_agent.name,
            "response": response
        })
        
        return f"[{selected_agent.name}]: {response}"
    
    def get_system_status(self) -> str:
        """Gibt System-Status zurück"""
        status = f"🤖 Multi-Agent System Status:\n"
        status += f"- Verfügbare Agents: {len(self.agents)}\n"
        
        for agent in self.agents:
            status += f"  • {agent.name} ({agent.specialization})\n"
        
        status += f"- Verarbeitete Anfragen: {len(self.conversation_history)}\n"
        
        return status

# Multi-Agent System aufbauen
orchestrator = AgentOrchestrator(llm)

# Spezialisierte Agents erstellen
math_agent = SpecializedAgent(
    name="MathBot",
    specialization="math", 
    tools=[CalculatorTool()],
    llm=llm
)

data_agent = SpecializedAgent(
    name="DataBot",
    specialization="data",
    tools=[DatabaseTool()],
    llm=llm
)

task_agent = SpecializedAgent(
    name="TaskBot", 
    specialization="tasks",
    tools=[TaskManagerTool()],
    llm=llm
)

# Agents hinzufügen
orchestrator.add_agent(math_agent)
orchestrator.add_agent(data_agent)
orchestrator.add_agent(task_agent)

print("\n=== Multi-Agent System Demo ===")
print(orchestrator.get_system_status())

# Tests
queries = [
    "Berechne 25 * 17 + 33",
    "Zeige mir alle Projekte",
    "Erstelle Aufgabe 'Meeting vorbereiten'",
    "Finde Developer in der Datenbank"
]

for query in queries:
    print(f"\n📝 Query: {query}")
    response = orchestrator.route_query(query)
    print(f"💬 Response: {response}")
```

## 5. Error Handling und Agent Monitoring

### Robuste Agent-Implementierung
```python
import time
import logging
from typing import Dict, List, Any

class MonitoredAgent:
    """Agent mit Monitoring und Error Handling"""
    
    def __init__(self, tools: List, llm, name: str = "Agent"):
        self.tools = tools
        self.llm = llm
        self.name = name
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0,
            "tool_usage": {},
            "error_log": []
        }
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=5,
            early_stopping_method="generate"
        )
    
    def run_with_monitoring(self, query: str, timeout: int = 30) -> Dict[str, Any]:
        """Führt Query mit umfassendem Monitoring aus"""
        
        start_time = time.time()
        self.metrics["total_queries"] += 1
        
        result = {
            "query": query,
            "response": None,
            "success": False,
            "execution_time": 0,
            "tools_used": [],
            "error": None
        }
        
        try:
            # Timeout-Überwachung (vereinfacht)
            response = self.agent.run(query)
            
            result["response"] = response
            result["success"] = True
            self.metrics["successful_queries"] += 1
            
            # Tool-Usage extrahieren (vereinfacht)
            # In Realität: aus Agent-Logs extrahieren
            result["tools_used"] = self._extract_tools_used(query)
            
        except Exception as e:
            error_msg = str(e)
            result["error"] = error_msg
            result["success"] = False
            self.metrics["failed_queries"] += 1
            
            # Error logging
            error_entry = {
                "timestamp": time.time(),
                "query": query,
                "error": error_msg,
                "error_type": type(e).__name__
            }
            self.metrics["error_log"].append(error_entry)
            
            # Fallback response
            result["response"] = f"❌ Entschuldigung, ich konnte Ihre Anfrage nicht bearbeiten: {error_msg}"
        
        finally:
            # Metriken aktualisieren
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            
            # Durchschnittliche Response Time berechnen
            total_time = self.metrics["avg_response_time"] * (self.metrics["total_queries"] - 1) + execution_time
            self.metrics["avg_response_time"] = total_time / self.metrics["total_queries"]
            
            # Tool usage aktualisieren
            for tool in result["tools_used"]:
                self.metrics["tool_usage"][tool] = self.metrics["tool_usage"].get(tool, 0) + 1
        
        return result
    
    def _extract_tools_used(self, query: str) -> List[str]:
        """Extrahiert verwendete Tools (vereinfacht)"""
        tools_used = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["berechne", "rechne", "mathematik"]):
            tools_used.append("Calculator")
        
        if any(word in query_lower for word in ["daten", "user", "projekt"]):
            tools_used.append("Database")
            
        if any(word in query_lower for word in ["aufgabe", "task", "todo"]):
            tools_used.append("TaskManager")
        
        return tools_used
    
    def get_performance_report(self) -> str:
        """Erstellt Performance-Bericht"""
        
        success_rate = (self.metrics["successful_queries"] / max(self.metrics["total_queries"], 1)) * 100
        
        report = f"""
🤖 {self.name} Performance Report
{'='*50}

📊 Statistiken:
- Total Queries: {self.metrics['total_queries']}
- Erfolgreiche Queries: {self.metrics['successful_queries']}
- Fehlgeschlagene Queries: {self.metrics['failed_queries']}
- Erfolgsrate: {success_rate:.1f}%
- Durchschnittliche Response Time: {self.metrics['avg_response_time']:.2f}s

🛠️ Tool Usage:
"""
        
        for tool, count in self.metrics["tool_usage"].items():
            percentage = (count / max(self.metrics["total_queries"], 1)) * 100
            report += f"- {tool}: {count} mal ({percentage:.1f}%)\n"
        
        if self.metrics["error_log"]:
            report += f"\n❌ Letzte Errors (max. 5):\n"
            for error in self.metrics["error_log"][-5:]:
                report += f"- {error['error_type']}: {error['error'][:100]}...\n"
        
        return report
    
    def reset_metrics(self):
        """Setzt Metriken zurück"""
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0, 
            "failed_queries": 0,
            "avg_response_time": 0,
            "tool_usage": {},
            "error_log": []
        }
        print(f"📊 Metriken für {self.name} zurückgesetzt")

# Monitored Agent testen
monitored_agent = MonitoredAgent(
    tools=[CalculatorTool(), DatabaseTool(), TaskManagerTool()],
    llm=llm,
    name="ProductionAgent"
)

print("=== Monitored Agent Demo ===")

# Test-Queries
test_queries = [
    "Berechne 15 * 23 + 7",
    "Zeige mir alle Developer",
    "Erstelle Aufgabe 'Code Review'",
    "Ungültige Query um Error zu testen xyz123"
]

for query in test_queries:
    print(f"\n📝 Query: {query}")
    result = monitored_agent.run_with_monitoring(query)
    
    print(f"✅ Success: {result['success']}")
    print(f"⏱️ Time: {result['execution_time']:.2f}s")
    print(f"🛠️ Tools: {result['tools_used']}")
    print(f"💬 Response: {result['response'][:100]}...")

# Performance Report
print(f"\n{monitored_agent.get_performance_report()}")
```

## ✅ Agent Mastery Checklist

- [ ] Basis Agents (ReAct, Zero-Shot) verstanden
- [ ] Custom Tools entwickelt und implementiert
- [ ] Agent Memory und State Management
- [ ] Multi-Agent Systeme aufgebaut
- [ ] Error Handling und Monitoring implementiert
- [ ] Tool-Integration gemeistert
- [ ] Agent Performance optimiert
- [ ] Production-ready Agent erstellt

## 🎯 Best Practices

1. **Tool Design**: Tools sollen spezifisch und gut dokumentiert sein
2. **Error Handling**: Immer Fallbacks für gescheiterte Tool-Calls
3. **Performance**: Monitoring und Optimierung der Agent-Performance
4. **Security**: Validierung aller Tool-Inputs
5. **Modularität**: Agents und Tools wiederverwendbar gestalten

## 🚀 Nächste Schritte

Jetzt bist du bereit für komplexe Agent-Systeme! Agents sind besonders mächtig, wenn sie mit RAG, Memory und externen APIs kombiniert werden.

**Nächstes Modul:** `10-tools` - Eigene Tools für Agents entwickeln
