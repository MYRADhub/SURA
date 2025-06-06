from pydantic import BaseModel

class OpenAIResponse(BaseModel):
    choose_direction: str # YES or NO
    target_goal: str # e.g. "A" for goal A, "B" for goal B, etc.
    explanation: str # 1-2 sentences explaining the choice