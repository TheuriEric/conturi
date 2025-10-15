from langchain_groq  import ChatGroq
import os
from dotenv import load_dotenv
from .components import logger


load_dotenv()
logger.info("Loaded environment variables")

try:
    router_llm = ChatGroq(
        model=os.getenv("GROQ_MODEL"),
        api_key=os.getenv("GROQ_API_KEY")
    )
    logger.info("Successfully connected to the router LLM")
except Exception as e:
    logger.exception("Failed to connect to the router LLM")

