from dotenv import load_dotenv
from .components import logger, router, Assistant
from crewai import Crew, Process
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from .crew import SynqCrew
import json
import os
import jwt
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Synq AI")



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
    crew_instance = SynqCrew()
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
async def run_synq_orchestration(user_query: str, crew_instance: Any) -> str:
    """
    Handles the dynamic three-phase execution logic:
    1. Main Agent determines the required worker agents.
    2. Runs only the required worker agents + the Presentation Agent.
    """
    if not crew_instance:
        logger.error("SynqCrew factory instance is not available.")
        return "System error: Orchestration factory not initialized."

    # Retrieve all tasks and agents from the factory/loader
    try:
        agent_task_map = crew_instance.get_agent_and_task_map()
    except AttributeError:
        logger.error("Crew instance does not have get_agent_and_task_map method.")
        return "System error: Failed to load agent map."

    # --- PHASE 1: ORCHESTRATION (Determining Route) ---
    logger.info("Phase 1: Running Main Orchestration Agent...")
    
    main_agent = agent_task_map["main_agent"]["agent"]
    main_task = agent_task_map["main_agent"]["task"]
    
    # 🔥 FIX: Initialize task.config and description if they are missing or None 🔥
    if main_task.config is None:
        main_task.config = {}
    if "description" not in main_task.config:
        # Use the original task description as the base
        main_task.config["description"] = main_task.description
        
    main_task.agent = main_agent
    
    # Inject the user query into the main task's description for the LLM to process
    main_task.config["description"] += f"\n\nUSER QUERY: {user_query}"
    
    # Create a minimal crew to run ONLY the main agent
    orchestration_crew = Crew(
        agents=[main_agent], 
        tasks=[main_task], 
        process=Process.sequential,
        verbose=False 
    )

    # Execute the orchestration
    try:
        orchestration_result = orchestration_crew.kickoff()
        # Assume the main agent's output is the desired JSON string
        import json
        result_str = str(orchestration_result)
        router_output = json.loads(result_str)
        required_agents: List[str] = router_output.get("required_agents", [])
    except Exception as e:
        logger.error(f"Main Orchestration Agent failed: {e}")
        return "An error occurred during routing. Please try again later."


    # --- PHASE 2: EXECUTION (Running Workers) ---
    if not required_agents:
        return "We couldn't determine a clear intent (events, career, or networking) from your query. Could you please clarify what you're looking for?"

    logger.info(f"Phase 2: Required Agents detected: {required_agents}")
    
    execution_agents = []
    execution_tasks = []

    # 1. Build the list of required tasks
    for name in required_agents:
        if name in agent_task_map and name != "main_agent":
            task_instance = agent_task_map[name]["task"]
            task_instance.agent = agent_task_map[name]["agent"]

            # 🔥 FIX: Ensure worker task config is initialized safely 🔥
            if task_instance.config is None:
                task_instance.config = {}
            if "description" not in task_instance.config:
                task_instance.config["description"] = task_instance.description
            
            # Pass the original query to the worker agents for context
            task_instance.config["description"] += f"\n\nCONTEXT/QUERY: {user_query}"
            
            execution_tasks.append(task_instance)
            execution_agents.append(task_instance.agent)
            
    # 2. Add the final Presentation Task (Refinement Phase)
    presentation_agent = agent_task_map["presentation_agent"]["agent"]
    presentation_task = agent_task_map["presentation_agent"]["task"]
    
    # 🔥 FIX: Ensure presentation task config is initialized safely 🔥
    if presentation_task.config is None:
        presentation_task.config = {}
    if "description" not in presentation_task.config:
        presentation_task.config["description"] = presentation_task.description

    presentation_task.agent = presentation_agent
    execution_tasks.append(presentation_task)
    execution_agents.append(presentation_agent)
    
    # Remove duplicates and run the execution crew
    if not execution_tasks:
        return "Routing was successful, but no relevant worker tasks were generated."
        
    execution_crew = Crew(
        agents=list(set(execution_agents)),
        tasks=execution_tasks,
        process=Process.sequential, # Use sequential to guarantee Presentation runs last
        verbose=False
    )
    
    # --- PHASE 3: RESULT (Final Output) ---
    logger.info("Phase 3: Running Execution and Presentation Crew...")
    final_result = execution_crew.kickoff()
    
    return final_result

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
                    # --- FIX: Pass crew_instance ---
                    crew_result = await run_synq_orchestration(user_query=query, crew_instance=crew_instance)
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
                # --- FIX: Pass crew_instance ---
                crew_result = await run_synq_orchestration(user_query=query, crew_instance=crew_instance)
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