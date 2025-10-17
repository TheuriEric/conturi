from pydantic import BaseModel, Field
from typing import Dict, Optional, List

class EventRequest(BaseModel):
    location: str = Field(..., description="Preferred event location")
    price: str = Field(..., description="User's budget range")
    event_description: str = Field(..., description="The user's description of the type of event they would want to attend")

class EventOutput(BaseModel):
    event_name: str = Field(..., description="Event name/title")
    location: str = Field(..., description="Event location")
    date: Optional[str] = Field(None, description="Event date if available")  # NEW
    price: Optional[str] = Field(None, description="Ticket price if known")   # NEW
    link: Optional[str] = Field(None, description="Registration link")        # NEW
    event_summary: str = Field(..., description="Brief description of the event, what it entails, who'll be there, and highlights")
    fun_fact: Optional[str] = Field(None, description="A fun fact about the event, theme, or anything related")  # Made optional


class PersonaProfiles(BaseModel):
    name: str = Field(..., description="The name and title of the person")
    role: str = Field(..., description="A brief description of their role and persona in general")
    relevance_score: Optional[float] = Field(None, description="0.0-1.0 score of relevance")  # NEW

class UserPersona(BaseModel):
    name: str = Field(..., description="User's name for personalization")
    age: str = Field(..., description="User's age")
    year_of_study: str = Field(..., description="User's year of study")
    brief_summary: str = Field(..., description="User's persona summary")
    
class JobDetails(BaseModel):
    position: str = Field(..., description="Role title")
    company: str = Field(..., description="Company name")  # NEW - was missing!
    location: str = Field(..., description="The company location")
    role_summary: str = Field(..., description="A brief explanation of the role")
    link: Optional[str] = Field(None, description="Application link")  # NEW

class CareerOutput(BaseModel):
    openings: List[JobDetails] = Field(default_factory=list, description="List of available positions")

class N8n(BaseModel):
    email: str = Field(..., description="User's email")
    schedule: str = Field(..., description="Does the user want to be sent emails of the outputs")
    prompt: str = Field(..., description="How does the user want the email to be sent? Any special requests/formatting?")

class IntentResponse(BaseModel):
    """Structured response from the Intent Analyst."""
    action: str = Field(
        description="The immediate next action. Must be 'handover' if the request is complete and ready for specialized execution, or 'response' if further conversation/context is needed."
    )
    message: str = Field(
        description="The full, conversational text response to the user. If action is 'handover', this message should confirm the task is complete and state it is being passed to the specialized agents."
    )