from dotenv import load_dotenv
from .components import logger, router
from fastapi import FastAPI, HTTPException



load_dotenv()
logger.info("Loaded environment variables")

app = FastAPI(
    title="Synq",
    description="Backend for Synq",
    version="1.0.0",  
)

@app.get("/")
async def root():
    return {"Message": "Visit /docs"}

@app.post("/chat")
async def synq(query: str):
    route = await router(user_query=query)
    if route == "langchain":
        pass
