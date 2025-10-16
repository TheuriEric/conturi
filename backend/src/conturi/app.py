from langchain_groq  import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from .components import logger, router


load_dotenv()
logger.info("Loaded environment variables")




