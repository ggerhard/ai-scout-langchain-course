# Intelligenter Chatbot - Praktisches Projekt

## 🤖 Projekt-Übersicht

In diesem Modul bauen wir einen vollständigen, intelligenten Chatbot mit:
- **Konversationsgedächtnis** (Memory)
- **Dokumenten-Wissen** (RAG) 
- **Werkzeuge** (Tools/Agents)
- **Web-Interface** (Streamlit)
- **Personalisierung** (User Profiles)

## 1. Chatbot-Architektur

### Core Components
```python
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool, AgentType
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class IntelligentChatbot:
    """Hauptklasse für intelligenten Chatbot"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.user_profiles = {}
        self.current_user = None
        self.setup_components()
        
    def setup_components(self):
        """Initialisiert alle Chatbot-Komponenten"""
        
        # LLM Setup
        self.llm = OpenAI(
            temperature=self.config.get('temperature', 0.7)
        )
        
        # Embeddings für RAG
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Memory Setup
        self.memory = ConversationBufferWindowMemory(
            k=self.config.get('memory_window', 10),
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Tools Setup
        self.tools = self.setup_tools()
        
        # Knowledge Base (RAG) Setup
        self.knowledge_base = self.setup_knowledge_base()
        
        # Hauptchain Setup
        self.setup_main_chain()
        
        print("✅ Chatbot erfolgreich initialisiert!")
    
    def setup_tools(self) -> List[Tool]:
        """Erstellt Tools für den Chatbot"""
        
        def get_current_time() -> str:
            """Gibt aktuelle Zeit zurück"""
            return f"Aktuelle Zeit: {datetime.now().strftime('%H:%M:%S, %d.%m.%Y')}"
        
        def calculate(expression: str) -> str:
            """Führt Berechnungen durch"""
            try:
                result = eval(expression.replace('^', '**'))
                return f"Ergebnis: {result}"
            except:
                return "Fehler bei der Berechnung"
        
        def save_note(note: str) -> str:
            """Speichert eine Notiz"""
            if not self.current_user:
                return "Bitte melde dich zuerst an"
            
            if self.current_user not in self.user_profiles:
                self.user_profiles[self.current_user] = {"notes": []}
            
            note_entry = {
                "text": note,
                "timestamp": datetime.now().isoformat(),
                "id": len(self.user_profiles[self.current_user].get("notes", []))
            }
            
            self.user_profiles[self.current_user].setdefault("notes", []).append(note_entry)
            return f"✅ Notiz gespeichert: {note[:50]}..."
        
        def get_notes() -> str:
            """Zeigt gespeicherte Notizen"""
            if not self.current_user or self.current_user not in self.user_profiles:
                return "Keine Notizen gefunden"
            
            notes = self.user_profiles[self.current_user].get("notes", [])
            if not notes:
                return "Keine Notizen vorhanden"
            
            note_list = []
            for note in notes[-5:]:  # Letzten 5 Notizen
                timestamp = datetime.fromisoformat(note["timestamp"]).strftime("%d.%m %H:%M")
                note_list.append(f"[{timestamp}] {note['text'][:100]}...")
            
            return "Deine letzten Notizen:\n" + "\n".join(note_list)
        
        def search_web(query: str) -> str:
            """Simuliert Web-Suche"""
            return f"🌐 Web-Suche für '{query}': Hier sind relevante Informationen aus dem Internet..."
        
        tools = [
            Tool(
                name="Zeit",
                func=get_current_time,
                description="Gibt die aktuelle Zeit und das Datum zurück"
            ),
            Tool(
                name="Rechner",
                func=calculate,
                description="Führt mathematische Berechnungen durch"
            ),
            Tool(
                name="Notiz_Speichern",
                func=save_note,
                description="Speichert eine Notiz für später"
            ),
            Tool(
                name="Notizen_Anzeigen",
                func=get_notes,
                description="Zeigt gespeicherte Notizen an"
            ),
            Tool(
                name="Web_Suche",
                func=search_web,
                description="Sucht aktuelle Informationen im Internet"
            )
        ]
        
        return tools
    
    def setup_knowledge_base(self) -> Optional[FAISS]:
        """Erstellt Knowledge Base aus Dokumenten"""
        docs_path = self.config.get('docs_path', './docs')
        
        if not os.path.exists(docs_path):
            os.makedirs(docs_path)
            # Erstelle Beispiel-Dokument
            with open(f"{docs_path}/chatbot_info.txt", "w", encoding="utf-8") as f:
                f.write("""
Chatbot Hilfe und Informationen

Dieser intelligente Chatbot kann dir bei verschiedenen Aufgaben helfen:

1. Fragen beantworten basierend auf dem Wissen in den Dokumenten
2. Berechnungen durchführen
3. Notizen speichern und verwalten
4. Aktuelle Zeit und Datum anzeigen
5. Web-Suchen simulieren

Der Chatbot merkt sich unsere Unterhaltung und kann auf vorherige Nachrichten Bezug nehmen.

Verfügbare Kommandos:
- "Speichere Notiz: [dein text]" - Speichert eine Notiz
- "Zeige meine Notizen" - Zeigt gespeicherte Notizen
- "Wie spät ist es?" - Aktuelle Zeit
- "Berechne 15 * 23" - Mathematische Berechnungen
- "Suche im Web nach [thema]" - Web-Suche

Der Chatbot ist personalisiert - melde dich mit deinem Namen an für bessere Erfahrung!
                """)
        
        try:
            # Dokumente laden
            loader = DirectoryLoader(
                docs_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
            
            if not documents:
                print("⚠️ Keine Dokumente gefunden")
                return None
            
            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Vector Store erstellen
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            print(f"✅ Knowledge Base erstellt mit {len(chunks)} Text-Chunks")
            return vectorstore
            
        except Exception as e:
            print(f"❌ Fehler beim Erstellen der Knowledge Base: {e}")
            return None
    
    def setup_main_chain(self):
        """Erstellt die Hauptchain für den Chatbot"""
        
        if self.knowledge_base:
            # RAG Chain mit Knowledge Base
            self.main_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.knowledge_base.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
        else:
            # Fallback: Simple Conversation Chain
            from langchain import ConversationChain
            self.main_chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=False
            )
        
        # Agent für Tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=3
        )
    
    def set_user(self, username: str):
        """Setzt aktuellen Benutzer"""
        self.current_user = username
        
        # User Profile initialisieren
        if username not in self.user_profiles:
            self.user_profiles[username] = {
                "created": datetime.now().isoformat(),
                "preferences": {},
                "notes": [],
                "conversation_count": 0
            }
        
        print(f"👤 Benutzer gesetzt: {username}")
    
    def needs_tool(self, message: str) -> bool:
        """Prüft ob eine Nachricht Tools benötigt"""
        tool_indicators = [
            "wie spät", "uhrzeit", "datum", "zeit",
            "berechne", "rechne", "plus", "minus", "mal", "*", "/", "+", "-",
            "speichere notiz", "notiz speichern", "merke dir",
            "zeige notizen", "meine notizen",
            "suche", "google", "web"
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in tool_indicators)
    
    def chat(self, message: str) -> Dict[str, any]:
        """Hauptmethode für Chat-Interaktion"""
        
        # User conversation count erhöhen
        if self.current_user and self.current_user in self.user_profiles:
            self.user_profiles[self.current_user]["conversation_count"] += 1
        
        response_data = {
            "message": message,
            "response": "",
            "sources": [],
            "tools_used": [],
            "timestamp": datetime.now().isoformat(),
            "user": self.current_user
        }
        
        try:
            # Entscheide zwischen Tool-Usage und Knowledge-Chat
            if self.needs_tool(message):
                # Verwende Agent mit Tools
                response = self.agent.run(message)
                response_data["tools_used"] = self.extract_tools_used(message)
                response_data["response"] = response
                
            else:
                # Verwende RAG/Knowledge Chain
                if hasattr(self.main_chain, 'retriever'):  # RAG Chain
                    result = self.main_chain({"question": message})
                    response_data["response"] = result["answer"]
                    
                    # Source documents hinzufügen
                    if "source_documents" in result:
                        response_data["sources"] = [
                            doc.page_content[:200] + "..." 
                            for doc in result["source_documents"][:2]
                        ]
                else:  # Simple Conversation Chain
                    response = self.main_chain.predict(input=message)
                    response_data["response"] = response
            
            # Personalisierung hinzufügen
            if self.current_user:
                response_data["response"] = self.personalize_response(
                    response_data["response"], message
                )
            
        except Exception as e:
            response_data["response"] = f"❌ Entschuldigung, es gab einen Fehler: {str(e)}"
            print(f"Chat Error: {e}")
        
        return response_data
    
    def extract_tools_used(self, message: str) -> List[str]:
        """Extrahiert verwendete Tools"""
        tools = []
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["zeit", "uhrzeit", "spät"]):
            tools.append("Zeit")
        if any(word in message_lower for word in ["berechne", "rechne", "*", "+"]):
            tools.append("Rechner")
        if "notiz" in message_lower:
            if "speicher" in message_lower:
                tools.append("Notiz_Speichern")
            else:
                tools.append("Notizen_Anzeigen")
        if "suche" in message_lower:
            tools.append("Web_Suche")
            
        return tools
    
    def personalize_response(self, response: str, message: str) -> str:
        """Fügt Personalisierung zur Antwort hinzu"""
        if not self.current_user:
            return response
        
        user_data = self.user_profiles.get(self.current_user, {})
        conv_count = user_data.get("conversation_count", 0)
        
        # Begrüßung für neue User
        if conv_count <= 3:
            greeting = f"Hallo {self.current_user}! "
            if not response.startswith(greeting):
                response = greeting + response
        
        # Gelegentliche personalisierte Notizen
        elif conv_count % 10 == 0:
            response += f"\n\n💡 Übrigens {self.current_user}, das ist unsere {conv_count}. Unterhaltung!"
        
        return response
    
    def get_user_stats(self) -> Dict:
        """Gibt User-Statistiken zurück"""
        if not self.current_user or self.current_user not in self.user_profiles:
            return {}
        
        user_data = self.user_profiles[self.current_user]
        return {
            "username": self.current_user,
            "conversations": user_data.get("conversation_count", 0),
            "notes_count": len(user_data.get("notes", [])),
            "member_since": user_data.get("created", "unbekannt"),
            "memory_messages": len(self.memory.chat_memory.messages) if hasattr(self.memory, 'chat_memory') else 0
        }

# Streamlit Web Interface
def create_streamlit_interface():
    """Erstellt Streamlit Web-Interface für den Chatbot"""
    
    st.set_page_config(
        page_title="Intelligenter Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Intelligenter Chatbot")
    st.write("Ein LangChain-basierter Chatbot mit Memory, RAG und Tools")
    
    # Sidebar für Konfiguration
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        # User Login
        username = st.text_input("Benutzername:", value="Gast")
        
        # Chatbot Settings
        temperature = st.slider("Kreativität (Temperature):", 0.0, 1.0, 0.7, 0.1)
        memory_window = st.slider("Memory Window:", 5, 20, 10)
        
        # Reset Button
        if st.button("🔄 Chat zurücksetzen"):
            st.session_state.clear()
            st.success("Chat zurückgesetzt!")
    
    # Chatbot initialisieren
    @st.cache_resource
    def init_chatbot():
        config = {
            'temperature': temperature,
            'memory_window': memory_window,
            'docs_path': './docs'
        }
        return IntelligentChatbot(config)
    
    chatbot = init_chatbot()
    chatbot.set_user(username)
    
    # Chat History in Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # User Stats anzeigen
    stats = chatbot.get_user_stats()
    if stats:
        st.sidebar.markdown("---")
        st.sidebar.subheader(f"👤 {stats['username']}")
        st.sidebar.write(f"📊 Gespräche: {stats['conversations']}")
        st.sidebar.write(f"📝 Notizen: {stats['notes_count']}")
        st.sidebar.write(f"🧠 Memory: {stats['memory_messages']} Nachrichten")
    
    # Chat Interface
    chat_container = st.container()
    
    # Chat History anzeigen
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Zusätzliche Infos bei Bot-Nachrichten
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    
                    if metadata.get("tools_used"):
                        st.caption(f"🛠️ Tools: {', '.join(metadata['tools_used'])}")
                    
                    if metadata.get("sources"):
                        with st.expander("📚 Quellen"):
                            for i, source in enumerate(metadata["sources"], 1):
                                st.write(f"{i}. {source}")
    
    # Chat Input
    if prompt := st.chat_input("Schreibe eine Nachricht..."):
        
        # User Message hinzufügen
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Bot Response
        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                response_data = chatbot.chat(prompt)
            
            st.write(response_data["response"])
            
            # Zusätzliche Infos anzeigen
            if response_data.get("tools_used"):
                st.caption(f"🛠️ Tools verwendet: {', '.join(response_data['tools_used'])}")
            
            if response_data.get("sources"):
                with st.expander("📚 Quellen anzeigen"):
                    for i, source in enumerate(response_data["sources"], 1):
                        st.write(f"**Quelle {i}:**")
                        st.write(source)
            
            # Bot Message zu History hinzufügen
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_data["response"],
                "metadata": {
                    "tools_used": response_data.get("tools_used", []),
                    "sources": response_data.get("sources", []),
                    "timestamp": response_data.get("timestamp")
                }
            })
    
    # Beispiel-Nachrichten
    st.markdown("---")
    st.subheader("💡 Beispiel-Nachrichten:")
    
    examples = [
        "Wie kann mir der Chatbot helfen?",
        "Wie spät ist es gerade?", 
        "Berechne 25 * 17 + 33",
        "Speichere Notiz: Meeting morgen um 10 Uhr",
        "Zeige meine Notizen",
        "Suche im Web nach LangChain Tutorial"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}"):
                # Simuliere Nachricht
                st.session_state.messages.append({"role": "user", "content": example})
                
                response_data = chatbot.chat(example)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["response"],
                    "metadata": {
                        "tools_used": response_data.get("tools_used", []),
                        "sources": response_data.get("sources", [])
                    }
                })
                st.experimental_rerun()

# Hauptprogramm
if __name__ == "__main__":
    
    # Kommandozeilen-Version
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        print("🤖 Intelligenter Chatbot (CLI-Modus)")
        print("=" * 50)
        
        config = {
            'temperature': 0.7,
            'memory_window': 10,
            'docs_path': './docs'
        }
        
        chatbot = IntelligentChatbot(config)
        
        username = input("Dein Name: ").strip() or "Gast"
        chatbot.set_user(username)
        
        print(f"\nHallo {username}! Ich bin dein intelligenter Assistent.")
        print("Schreibe 'quit' zum Beenden.\n")
        
        while True:
            user_input = input(f"{username}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Auf Wiedersehen!")
                break
            
            if user_input:
                response_data = chatbot.chat(user_input)
                print(f"🤖: {response_data['response']}")
                
                if response_data.get("tools_used"):
                    print(f"   🛠️ Tools: {', '.join(response_data['tools_used'])}")
                
                print()
    
    else:
        # Streamlit-Version
        create_streamlit_interface()

"""
Installation und Setup:

1. Erforderliche Pakete installieren:
   pip install streamlit langchain openai faiss-cpu sentence-transformers

2. Docs-Ordner erstellen und Dokumente hinzufügen:
   mkdir docs
   # Füge .txt Dateien hinzu

3. OpenAI API Key setzen:
   export OPENAI_API_KEY="dein-api-key"

4. Starten:
   streamlit run chatbot.py        # Web-Interface
   python chatbot.py --cli         # Kommandozeile

Features:
✅ Konversationsgedächtnis
✅ RAG (Dokumenten-basiertes Wissen)
✅ Tools (Zeit, Rechner, Notizen, Web-Suche)
✅ Benutzer-Profile
✅ Web und CLI Interface
✅ Personalisierte Antworten
✅ Source-Tracking
"""
