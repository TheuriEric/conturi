import logging
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.memory import ConversationBufferMemory
from .chat_models import IntentResponse
import os
from pathlib import Path
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from typing import Optional
from upstash_redis import Redis
import requests
import os
import asyncio
import time
import json

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))


try:
    project_root = Path(__file__).resolve().parent.parent.parent
    pdf_path = f"{project_root}/knowledge"
    persist_directory = f"{project_root}/db"
    logger.info("Registered required directories")
except Exception as e:
    logger.exception("Failed to register required directories")

try:
    embeddings = CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="embed-english-v3.0"
    )
    logger.info("Successfully loaded the embedding model: Cohere")
except Exception as e:
    logger.exception("Failed to load Cohere's embedding model")
    raise e

try:
    logger.info("Loading the vector database")
    knowledge_base = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    ).as_retriever()
    logger.info("Successfully loaded the vector daabase")
except Exception as e:
    logger.exception("Failed to load the vector database")


class ContentRetriever():
    def rag_tool(self, query: str) -> str:
        """
        A tool to retrieve relevant context from the Chroma knowledge base."""
        try:
            logger.info(f"RAG Tool: Searching for documents related to the topic: '{query}'...")
            retrieved_docs = knowledge_base.invoke(query)
            context = "\n\n-------------\n\n".join([doc.page_content for doc in retrieved_docs])
            
            if not context:
                logger.warning("RAG Tool: No relevant information found.")
                return "No relevant information found in the knowledge base."
            
            logger.info(f"RAG Tool: Successfully retrieved context.")
            return context
        except Exception as e:
            logger.error(f"RAG Tool: Error during retrieval - {e}", exc_info=True)
            return f"Error retrieving context from knowledge base: {e}. Please proceed without."
        
    def web_search_tool(self,query: str) -> str:
        """
        Searches the web using DuckDuckGo and extracts readable content
        from top search results.
        """
        try:
            logger.info(f"Web Search Tool: Searching for '{query}'...")
            results_text = ""

            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query)]

            if not results:
                return "No results found for your query."

            logger.info(f"Web Search Tool: Retrieved {len(results)} search results.")
            articles = []

            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                link = result.get("href", None)
                snippet = result.get("body", "")
                if not link:
                    continue

                logger.info(f"Fetching article {i}: {title} ({link})")

                try:
                    response = requests.get(link, timeout=8)
                    soup = BeautifulSoup(response.text, "html.parser")

                    paragraphs = [p.get_text() for p in soup.find_all("p")]
                    article_text = "\n".join(paragraphs[:5])  # first few paragraphs only

                    articles.append(f"### {title}\n🔗 {link}\n\n{article_text}\n")
                except Exception as e:
                    logger.warning(f"Skipping {link}: {e}")
                    continue

            if not articles:
                logger.info("Returned no articles from the search results")
                return "No readable articles found from the search results."
            

            results_text = "\n\n---\n\n".join(articles)
            logger.info("Web Search Tool: Successfully extracted article content.")
            return results_text

        except Exception as e:
            logger.error(f"Web Search Tool failed: {e}", exc_info=True)
            return f"Error searching the web: {e}"

SYSTEM_CAPABILITIES = """
Synq has the following specialized services (agents) and their data requirements.
Use these to guide the user to provide all necessary context:

1.  **Event Discovery Curator:** To find events, we ideally need to know the **Location, Budget/Price Range, and Event Vibe/Type** (e.g., 'tech meetup', 'chill nature escape').
2.  **Professional Networking Analyst:** To find professionals/companies, we need to know a **Specific Event Name** or a **Theme/Industry and Location** (e.g., 'solar energy leaders in Kenya').
3.  **Career Intelligence Analyst:** To find opportunities, we need to know the user's **Area of Study/Specialization, Preferred Location, and Desired Type of Role** (e.g., 'internship', 'graduate job').

General questions (like 'Who are you?') do not require special data.
"""
async def router(user_query: str) -> str:
    """Route queries intelligently between CrewAI or LangChain (general chat)."""
    try:
        router_llm = AIModels.router_llm()
        logger.info("Successfully connected to the router LLM")
    except Exception as e:
        logger.exception("Failed to connect to the router LLM")
    router_prompt = ChatPromptTemplate.from_template("""
    You are a routing expert for the Synq platform. Your job is to decide the immediate next step.

    System Capabilities: Synq can handle specialized tasks like **finding events, professional connections, or career opportunities** (Route: 'crewai'). All other queries, incomplete specialized requests, or conversational chat should go to the general chat handler (Route: 'langchain').

    Decision Logic:
    1. If the query is conversational, general knowledge, or requires asking for more details (like 'What events are happening?'), return **'langchain'**.
    2. If the query is a full, complete, and highly specific request that is ready for specialized execution (e.g., "Find me tech events in Nairobi this weekend with a budget under 3000."), return **'crewai'**.

    Respond with one word only: crewai or langchain.

    User query: {query}
    Response:
    """)
    router_chain = router_prompt | router_llm | StrOutputParser()
    try:
        logger.info("Determining route...")
        route = await router_chain.ainvoke({"query": user_query})
        if route not in ["crewai", "langchain"]:
            logger.warning(f"Router returned invalid route '{route}'. Defaulting to 'langchain'.")
            return "langchain"

        logger.info(f"Routing conversation to {route}")
        return route
    except Exception as e:
        logger.exception("Router failed to route the conversation. Proceeding with langchain")
        return "langchain"

class SimpleTracker:
    """Dead simple: just counts tool usage"""
    def __init__(self):
        self.count = {}
    
    def can_use(self, tool: str, max_uses: int = 3):
        return self.count.get(tool, 0) < max_uses
    
    def use(self, tool: str):
        self.count[tool] = self.count.get(tool, 0) + 1
    
    def reset(self):
        self.count = {}


class AIModels():
    @staticmethod
    def router_llm():
        from langchain_groq import ChatGroq
        try:
            logger.info("Connecting to the router LLM")
            router_llm =  ChatGroq(model=os.getenv("ROUTER_LLM"),
                                   api_key=os.getenv("GROQ_API_KEY"))
            logger.info("Connected to the LLM")
            return router_llm
        except Exception as e:
            logger.exception("Failed to connect to the initial router. Retrying...")
            try:
                logger.info("Connected to the general LLM")
                return AIModels.general_llm()
            except Exception as e2:
                logger.exception("Failed to connect to a routing LLM")
                raise e2
    @staticmethod
    def general_llm():
        from langchain_groq import ChatGroq
        try:
            logger.info("Connecting to the general chat LLM")
            general_llm = ChatGroq(
                model=os.getenv("GENERAL_MODEL"),
                api_key=os.getenv("GROQ_API_KEY")
            )
            logger.info("Connected to the LLM")
            return general_llm
        except Exception as e:
            logger.exception("Failed to connect to initial LLM. Reconnecting to another model")
            return e
    @staticmethod
    def crew_llm():
        from crewai import LLM
        try:
            logger.info("Connecting to the CrewAI llm")
            return LLM(
                model="gemini/gemini-2.5-pro",
                api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7
                )
        except Exception as e:
            logger.exception("Failed to connect to the initial LLM")
            try:
                return LLM(
                model="gemini/gemini-2.5-pro",
                api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.5
                )
            except Exception as e2:
                logger.exception("Fallback Gemini LLM also failed")
                raise e2

from langchain_core.runnables import RunnableMap
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
MESSAGE_EXPIRY_SECONDS = 86400

# Save this to prevent kuharibu mambo
# Save ya pili

class RedisChatMemory(BaseChatMessageHistory):
    """Persist chat sessions in Redis for long-term memory."""
    
    def __init__(self, session_id: str, session_expiry_seconds: int = MESSAGE_EXPIRY_SECONDS): 
        self.session_id = session_id # Store the session_id
        self.expiry = session_expiry_seconds

    def _get_messages_sync(self) -> list[BaseMessage]:
        """Synchronous message loading logic (for internal use and sync chains)."""
        raw = redis.get(self.session_id)
        if not raw:
            return []
        try:
            messages_data = json.loads(raw)
            messages = []
            for m in messages_data:
                if m["type"] == "human":
                    messages.append(HumanMessage(content=m["content"]))
                else:
                    messages.append(AIMessage(content=m["content"]))
            return messages
        except Exception as e:
            logger.warning(f"Failed to parse Redis history for session {self.session_id}: {e}")
            return []
            
    def _save_messages_sync(self, messages: list[BaseMessage]):
        """Synchronous message saving logic (for internal use and sync chains)."""
        try:
            serialized = []
            for m in messages:
                msg_type = "human" if isinstance(m, HumanMessage) else "ai"
                serialized.append({"type": msg_type, "content": m.content,"timestamp": int(time.time()) })
            redis.set(self.session_id, json.dumps(serialized), ex=self.expiry)
        except Exception as e:
            logger.error(f"Failed to save Redis history for session {self.session_id}: {e}")
    
    @property
    def messages(self) -> list[BaseMessage]: 
        """Load conversation history synchronously (required property)."""
        return self._get_messages_sync()
    
    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages synchronously."""
        current_messages = self._get_messages_sync()
        current_messages.extend(messages)
        self._save_messages_sync(current_messages)

    def add_ai_message(self, message: str) -> None:
        """Add an AI message synchronously."""
        current_messages = self._get_messages_sync()
        current_messages.append(AIMessage(content=message))
        self._save_messages_sync(current_messages)

    def clear(self) -> None:
        """Clear all messages synchronously."""
        redis.delete(self.session_id)

    # --- REQUIRED ASYNCHRONOUS METHODS (for ainvoke compatibility) ---
    async def aget_messages(self) -> list[BaseMessage]:
        """Load conversation history asynchronously (REQUIRED by ainvoke)."""
        return await asyncio.to_thread(self._get_messages_sync) # <-- Use asyncio.to_thread

    async def aadd_messages(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages asynchronously (REQUIRED by ainvoke)."""
        current_messages = await self.aget_messages()
        current_messages.extend(messages)
        await asyncio.to_thread(self._save_messages_sync, current_messages)
    
    async def aclear(self) -> None:
        """Clear all messages asynchronously."""
        await asyncio.to_thread(self.clear)

class Assistant():
    def __init__(self):
        self.retriver = ContentRetriever()
        self.general_llm = AIModels.general_llm()
        self.system_capabilities = SYSTEM_CAPABILITIES
        self.parser = JsonOutputParser(pydantic_object=IntentResponse)
        

        chat_template = ChatPromptTemplate.from_messages([
            ("system", f"""
            ## 🤖 Persona: Synq General Assistant (The Intent Analyst)

            **Role:** You are the **Synq General Assistant**—an intelligent, conversational, and highly human-like interface for the Synq platform. You are the first point of contact and must always maintain a professional, youthful, and helpful Synq brand tone.

            **Goal:**
            1. Handle all general chat and informational queries smoothly.
            2. For specialized requests (Events, Careers, Networking), you must act as the **Intent Analyst**—gathering all necessary information before signaling the CrewAI system.

            ---

            ## 🧠 Synq System Knowledge
            
            {self.system_capabilities}

            ---

            ## 💬 Conversation Instructions

            1. **Context Awareness:** Review the conversation history below and the RAG content provided to inform your response.
            2. **General Query Handling (Default):**
                * Your **primary domain** is **Synq** — the platform, its features, agents, event/career systems, memory, APIs, and architecture.
                * If a user asks a question **related to Synq**, its agents, RAG setup, architecture, or workflows — answer in full detail using your RAG knowledge base where possible.
                * If a user asks a question **not related to Synq** (e.g., personal questions, random topics, coding unrelated to Synq), respond briefly but **remind them politely** that:
                    > "I'm designed to assist mainly with Synq’s platform, systems, and services — would you like me to connect your question to Synq’s context?"
                * If possible, **rephrase** their unrelated question to show how it might connect to Synq’s ecosystem.

            3.  **Specialized Request Handling (The Pivot):**
                * If the user implies a need for a specialized service, check both: a) The Synq System Knowledge for required info and b) The Conversation History for information already provided.
                * If information is MISSING, set 'action' to **'response'** and ask concise, leading questions.
            4.  **Complete Request Handling (Final Action):** If the request is complete, set the 'action' field to **'handover'**. The 'message' must summarize the request and state that the task is being passed to the specialized agents. The 'user_request' should be a summary of the user's complete request, in first person...If the user is requesting for a service, you should summarize in a well formatted way for example: 
            - "I am looking for tech events in Nairobi that are taking place this week and the weekend and I do not have a specific budget range." 
            - "I am looking for entry-level accounting opportunities in Nairobi or even remote. If there is no final summary, just write "No final summary"
            5.  **Agent Masking:** Use the human-readable service title.

            ---
            
            ## 📤 Output Format Instructions
            Your final output MUST be a JSON object that conforms to the schema below.
            {{format_instructions}}

            ## RAG content
            {{rag_content}} 
            
            ---
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_query}"),
        ])
        self.chain = RunnableMap({
            "rag_content": lambda x: self.retriver.rag_tool(x["user_query"]),
            "user_query": lambda x: x["user_query"],
            "format_instructions": lambda x: self.parser.get_format_instructions(),
            "history": lambda x: x["history"],
        }) | chat_template | self.general_llm | self.parser

        
    
    async def langchain(self, user_query: str, session_id: str = "trial"):
        """
        Handles the main LangChain conversation for Synq general chat.
        """
        logger.info(f"Invoking main chain for query: {user_query} with Session ID: {session_id}")
        try:
            # Step 1: Get memory
            memory = RedisChatMemory(session_id=session_id)
            
            # Step 2: Load existing history from Redis
            history = await memory.aget_messages()
            logger.info(f"Loaded {len(history)} messages from history")
            
            # Step 3: Run the chain with history
            response = await self.chain.ainvoke({
                "user_query": user_query,
                "history": history
            })
            
            # Step 4: Save both user message and AI response to Redis
            logger.info("Saving conversation to Redis...")
            await memory.aadd_messages([
                HumanMessage(content=user_query),
                AIMessage(content=json.dumps(response))
            ])
            
            # Step 5: Verify save worked
            updated_history = await memory.aget_messages()
            logger.info(f"History now has {len(updated_history)} messages")
            
            logger.info("LangChain response generated successfully.")
            return response
        except Exception as e:
            logger.error("Error in Assistant.langchain", exc_info=True)
            raise e

