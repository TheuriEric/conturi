from pydantic import BaseModel, Field
from typing import Dict

class EventRequest(BaseModel):
    location: str = Field(..., description="Preferred event location")
    price: str = Field(..., description="User's budget range")
    event_description: str = Field(..., description="The user's description of the type of event they would want to attend")

class EventOutput(BaseModel):
    event_name: str = Field(..., description="Event name/title")
    location: str = Field(..., description="Event location")
    event_summary: str = Field(..., description="A brief description of the event..what it entails..who'll be there, price and the rest")
    fun_fact: str = Field(..., description="A fun fact about the event,theme or anything related to the event")

class PersonaProfiles(BaseModel):
    name: str = Field(..., description="The name and title of the person")
    role: str = Field(..., description="A brief description of their role and persona in general")

class UserPersona(BaseModel):
    name: str = Field(..., description="User's name for personalization")
    age: str = Field(..., description="User's age")
    year_of_study: str = Field(..., description="User's year of study")
    brief_summary: str = Field(..., description="User's persona summary")
    
class JobDetails(BaseModel):
    position: str = Field(..., description="Role title")
    location: str = Field(..., description="The company location")
    role_summary: str = Field(..., description="A brief explanation of the role")

class CareerOutput(BaseModel):
    openings: Dict[str, JobDetails] = Field(...,description="Summary of available related positions")

class N8n(BaseModel):
    email: str = Field(..., description="User's email")
    schedule: str = Field(..., description="Does the user want to be sent emails of the outputs")
    prompt: str = Field(..., description="How does the user want the email to be sent? Any special requests/formatting?")