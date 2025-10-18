from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import tool
from typing import List
from .components import ContentRetriever, project_root, logger, AIModels, SimpleTracker
tracker = SimpleTracker()
model = AIModels()
content_retriever = ContentRetriever()

@tool("rag_tool")
def rag_tool(query: str) -> str:
    """Retrieve relevant context from the internal knowledge base.
    
    LIMIT: Max 2 uses per task.
    """
    if not tracker.can_use('rag_tool', max_uses=2):
        return "RAG tool limit reached. Continue with available information."
    
    tracker.use('rag_tool')
    logger.info(f"RAG tool used ({tracker.count.get('rag_tool', 0)}/2)")
    
    try:
        return content_retriever.rag_tool(query=query)
    except Exception as e:
        logger.exception("RAG tool failed")
        return "RAG tool unavailable."

@tool("web_search_tool")
def web_search_tool(query: str) -> str:
    """Search the web for information. Returns readable summaries.
    
    LIMIT: Max 3 uses per task. Use strategically.
    BEST PRACTICE: Include date/location in query (e.g., "marketing events NYC October 2025")
    """
    if not tracker.can_use('web_search_tool', max_uses=3):
        return "Search limit reached (3/3). Work with information you have."
    
    tracker.use('web_search_tool')
    remaining = 3 - tracker.count.get('web_search_tool', 0)
    logger.info(f"Web search used ({tracker.count.get('web_search_tool', 0)}/3) - {remaining} remaining")
    
    try:
        result = content_retriever.web_search_tool(query)
        if remaining > 0:
            result += f"\n\n[System: {remaining} searches remaining]"
        return result
    except Exception as e:
        logger.exception("Web search failed")
        return "Web search unavailable."

@CrewBase
class SynqCrew():
    """Conturi crew"""
    def __init__(self):
        import os
        import yaml

        config_dir = f"{project_root}/src/conturi/config"


        try:
            with open(os.path.join(config_dir, "agents.yaml"), "r") as f:
                self.agents_config = yaml.safe_load(f)
                logger.info("Successfully loaded the agents description")
            with open(os.path.join(config_dir, "tasks.yaml"), "r") as f:
                self.tasks_config = yaml.safe_load(f)
                logger.info("Successfully loaded tasks descriprions")
            logger.info("Configuration files loaded successfully")
        except Exception as e:
            logger.warning("Failed to load config files: %s", e)
            self.agents_config = {}
            self.tasks_config = {}

        self.agents: List[BaseAgent] = []
        self.tasks: List[Task] = []


    @agent
    def main_agent(self) -> Agent:
        return Agent(config=self.agents_config["main_agent"],
                     llm=model.crew_llm(),verbose=True, max_iter=2)

    @agent
    def event_agent(self) -> Agent:
        return Agent(config=self.agents_config["event_agent"],
                     tools=[web_search_tool, rag_tool],llm=model.crew_llm(),
                       verbose=True, max_iter=3)

    @agent
    def professional_agent(self) -> Agent:
        return Agent(config=self.agents_config["professional_agent"],
                     tools=[web_search_tool, rag_tool],
                     llm=model.crew_llm(),
                    verbose=True,max_iter=3)

    @agent
    def career_agent(self) -> Agent:
        return Agent(config=self.agents_config["career_agent"],
                     tools=[web_search_tool, rag_tool],
                     llm=model.crew_llm(),
                    verbose=True, max_iter=3)

    @agent
    def presentation_agent(self) -> Agent:
        return Agent(config=self.agents_config["presentation_agent"],
                     llm=model.crew_llm(), verbose=True, max_iter=1)


   
    @task
    def main_task(self) -> Task:
        return Task(config=self.tasks_config["main_task"])

    @task
    def event_task(self) -> Task:
        return Task(config=self.tasks_config["event_task"])

    @task
    def professional_task(self) -> Task:
        return Task(config=self.tasks_config["professional_task"])

    @task
    def career_task(self) -> Task:
        return Task(config=self.tasks_config["career_task"])

    @task
    def presentation_task(self) -> Task:
        return Task(config=self.tasks_config["presentation_task"])

    @crew
    def crew(self) -> Crew:
        """Creates the Conturi crew"""
        try:
            logger.info("Building the Synq Crew")
            tracker.reset()
            agents = [
                self.main_agent(),
                self.event_agent(),
                self.professional_agent(),
                self.career_agent(),
                self.presentation_agent()
            ]
            tasks = [
                self.main_task(),
                self.event_task(),
                self.professional_task(),
                self.career_task(),
                self.presentation_task()
            ]
            return Crew(
                agents=agents, 
                tasks=tasks, 
                process=Process.sequential,
                verbose=True,
                max_rpm=15
            )
        except Exception as e:
            logger.exception("Failed to create Synq Crew")
            raise

if __name__ == "__main__":
    synq = SynqCrew()
    crew = synq.crew()
    result = crew.run()
    print(result)