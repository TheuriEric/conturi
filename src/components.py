import logging
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = f"{project_root}/knowledge"
    persist_directory = f"{project_root}/db"
    logger.info("Registered required directories")
except Exception as e:
    logger.exception("Failed to register required directories")

try:
    embeddings = CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE API KEY"),
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
    def get_content(self):
        pass