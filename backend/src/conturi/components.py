import logging
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq  import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path
from bs4 import BeautifulSoup
from ddgs import DDGS
import requests
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
        A tool to retrieve relevant context from the Pinecone knowledge base."""
        try:
            logger.info(f"RAG Tool: Searching for documents related to the topic: '{query}'...")
            retrieved_docs = knowledge_base.get_relevant_documents(query)
            context = "\n\n-------------\n\n".join([doc.page_content for doc in retrieved_docs])
            
            if not context:
                logger.warning("RAG Tool: No relevant information found.")
                return "No relevant information found in the knowledge base."
            
            logger.info(f"RAG Tool: Successfully retrieved context.")
            return context
        except Exception as e:
            logger.error(f"RAG Tool: Error during retrieval - {e}", exc_info=True)
            return f"Error retrieving context from knowledge base: {e}. Please proceed without."
        
    def web_search_tool(self,query: str, max_results: int = 5) -> str:
        """
        Searches the web using DuckDuckGo and extracts readable content
        from top search results. Ideal for generating blog posts.
        """
        try:
            logger.info(f"Web Search Tool: Searching for '{query}'...")
            results_text = ""

            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=max_results)]

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


       
async def router(user_query: str) -> str:
    """Route queries intelligently between CrewAI or LangChain (general chat)."""
    try:
        router_llm = ChatGroq(
            model=os.getenv("GROQ_MODEL"),
            api_key=os.getenv("GROQ_API_KEY")
        )
        logger.info("Successfully connected to the router LLM")
    except Exception as e:
        logger.exception("Failed to connect to the router LLM")
    router_prompt = ChatPromptTemplate.from_template("""
    You are a routing expert. 
    Decide whether to route the user query to 'crewai' (for content generation)
    or 'langchain' (for general chat, reasoning, and conversation).

    Respond with one word only: crewai or langchain.

    User query: "{query}"
    Response:
    """)
    router_chain = router_prompt | router_llm | StrOutputParser()
    try:
        logger.info("Determining route...")
        route = await router_chain.ainvoke({"query": user_query})
        logger.info(f"Routing conversation to {route}")
        return route.strip().lower()
    except Exception as e:
        logger.exception("Router failed to route the conversation. prpceeding with langchain")
        return "langchain"
