"""Pydantic schemas for structured LLM outputs."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ResolutionOutput(BaseModel):
    """Resolution Agent output schema."""

    is_exception: Literal["YES", "NO"] = Field(
        description="Whether this delivery event is a real actionable exception"
    )
    resolution: Literal[
        "RESCHEDULE", "REROUTE_TO_LOCKER", "REPLACE", "RETURN_TO_SENDER", "N/A"
    ] = Field(description="Resolution action. N/A if is_exception is NO")
    rationale: str = Field(
        description="Step-by-step reasoning for the classification and resolution decision"
    )

    @model_validator(mode="after")
    def validate_consistency(self):
        """Enforce that is_exception and resolution are mutually consistent."""
        if self.is_exception == "YES" and self.resolution == "N/A":
            raise ValueError("resolution cannot be N/A when is_exception is YES")
        if self.is_exception == "NO" and self.resolution != "N/A":
            raise ValueError("resolution must be N/A when is_exception is NO")
        return self


class CommunicationOutput(BaseModel):
    """Communication Agent output schema."""

    tone_label: Literal["FORMAL", "CASUAL"] = Field(
        description="Tone of the customer message, inferred from customer tier"
    )
    communication_message: str = Field(
        description="The customer-facing notification message"
    )


class CriticResolutionOutput(BaseModel):
    """Critic Agent - resolution validation output."""

    decision: Literal["ACCEPT", "ESCALATE", "REVISE"] = Field(
        description="ACCEPT: valid. ESCALATE: needs supervisor. REVISE: send back to Resolution Agent."
    )
    rationale: str = Field(description="Reasoning for the validation decision")


class CriticCommunicationOutput(BaseModel):
    """Critic Agent - communication validation output."""

    decision: Literal["ACCEPT", "ESCALATE"] = Field(
        description="ACCEPT: message is appropriate. ESCALATE: needs supervisor review."
    )
    rationale: str = Field(description="Reasoning for the validation decision")
