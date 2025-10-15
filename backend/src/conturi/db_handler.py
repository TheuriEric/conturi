from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader, WebBaseLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from .components import logger, pdf_path, persist_directory, embeddings

load_dotenv()
logger.info("Loaded environment variables")

try:
    resources_links = []
    web_loader = WebBaseLoader(
        resources_links    
    )
    logger.info("Loaded web based content. Ready for chunking")
except Exception as e:
    logger.exception("Failed to load web data.")
try:
    pdf_loader = DirectoryLoader(
        loader_cls=PDFPlumberLoader,
        glob="*.pdf",
        path=pdf_path
    )
    logger.info("Loaded the PDF content. Ready for chunking")
except Exception as e:
    logger.exception("Failed to load PDF data.")


try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=25)
    logger.info("Initialized the text splitter")
except Exception as e:
    logger.exception("Failed to initialize the text splitter")

try:
    logger.info("Starting to load the content to the vector database")
    web_content = web_loader.load()
    logger.info("Loaded the web content")
    pdf_content = pdf_loader.load()
    logger.info("Loaded the pdf content")
except Exception as e:
    logger.exception("Failed to load the content")

try:
    logger.info("Preparing to chunk data")
    web_chunks = text_splitter.split_documents(web_content)
    logger.info("Chunked web content")
    pdf_chunks = text_splitter.split_documents(pdf_content)
    logger.info("Chunked PDF content")
    all_chunks = web_chunks + pdf_chunks
except Exception as e:
    logger.exception("Failed to chunk the loaded data")

try:
    logger.info("Populating the vector database")
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        persist_directory=persist_directory,
        embedding=embeddings
    )
    logger.info("Successfully populated the vector database")
except Exception as e:
    logger.exception("Failed to populate the vector database")
    raise

