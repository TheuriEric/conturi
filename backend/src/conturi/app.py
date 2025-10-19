from dotenv import load_dotenv
from .components import logger, router, Assistant
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from .crew import SynqCrew
import os
import jwt



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
    crew_instance = None

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
async def synq(query: str = Body(..., embed=True)):
    final_response = "No response generated"
    status = "Error"
    session_id = "trial"
    route = await router(user_query=query)
    if route == "langchain":
        logger.info("Routing conversation to langchain")
        try:
            intent_response_obj:dict = await assistant.langchain(user_query=query, session_id=session_id)
            action = intent_response_obj.get("action", "").lower()
            if action == "handover" and crew_instance:
                    logger.info("Assistant signaled HANDOVER (Structured). Executing CrewAI...")
                    crew_result = crew_instance.kickoff(inputs={"user_query": query})
                    final_response = crew_result
                    status = "executed"
            else:
                final_response = intent_response_obj.get("message", "No response generated.")
                status = "conversational"
        except Exception as e:
            logger.exception("Error during langchain execution/handover")
            return {"response": "Apologies! An internal error occurred while connecting with the assistant. I couldn't process the response.", "status": "error"}

    elif route == "crewai":
        logger.info("Routing conversation to CrewAI directly")
        if not crew_instance:
            logger.error("CrewAI instance is not available.")
            final_response = "Specialized agents are offline. Please try again later."
            status = "error"
        else:
            try:
                crew_result = crew_instance.kickoff(inputs={"user_query":query})
                final_response = crew_result
                status = "executed"
            except Exception as e:
                logger.exception("Error executing CrewAI directly.")
                final_response = "Apologies! The specialized system encountered an error."
                status = "error"
    else:
        logger.warning(f"Router returned unhandled route: {route}")
        final_response = "I couldn't determine the next step for that request."
        status = "error"
    return {"response":final_response,
            "status": status}
        

