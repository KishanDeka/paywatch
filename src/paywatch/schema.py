import math
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    event_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    time: float = Field(ge=0)
    amount: float = Field(ge=0, le=1e12)
    v: list[float] = Field(min_length=28, max_length=28)

    @field_validator("v")
    @classmethod
    def finite_features(cls, values):
        if not all(math.isfinite(x) and abs(x) <= 1e6 for x in values):
            raise ValueError("V features must be finite and within numerical limits")
        return values
