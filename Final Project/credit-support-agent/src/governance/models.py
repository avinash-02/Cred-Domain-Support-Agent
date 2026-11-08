from pydantic import BaseModel

class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str
