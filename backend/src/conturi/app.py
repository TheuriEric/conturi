from dotenv import load_dotenv
from .components import logger, router, Assistant
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .crew import SynqCrew



load_dotenv()
logger.info("Loaded environment variables")

app = FastAPI(
    title="Synq",
    description="Backend for Synq",
    version="1.0.0",  
)
try:
    assistant = Assistant()
    logger.info("Initialized assistants")
except Exception as e:
    logger.exception("Failed to initialize an assistant")
try:
    crew_instance = SynqCrew().crew()
    logger.info("Successfully initialized crew instance")
except Exception as e:
    logger.exception("Failed to initialize the crew")
    raise e

origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"Message": "Visit /docs"}

@app.post("/chat")
async def synq(query: str):
    route = await router(user_query=query)
    if route == "langchain":
        logger.info("Routing conversation to langchain")
        response = await assistant.langchain(user_query=query,history="No previous history")
        return response
    elif route == "crewai":
        logger.info("Routing conversation to CrewAI")
    else:
        return False
        

