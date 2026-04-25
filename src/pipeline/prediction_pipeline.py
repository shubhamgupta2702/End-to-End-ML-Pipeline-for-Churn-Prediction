import sys
import pandas as pd
from src.exception.exception import CustomException
from src.logger.logger import logger
from pydantic import BaseModel, Field
from typing import Annotated, Optional


from pydantic import BaseModel, Field
from typing import Literal


class CustomerChurn(BaseModel):

    gender: Literal["Male", "Female"] = Field(..., description="Gender of the customer")

    SeniorCitizen: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indicates if the customer is a senior citizen (1 = Yes, 0 = No)",
    )

    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has a partner"
    )

    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has dependents"
    )

    tenure: int = Field(
        ...,
        ge=0,
        description="Number of months the customer has stayed with the company", examples=[1, 2, 3]
    )

    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has phone service", examples=["Yes", "No"]
    )

    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Whether the customer has multiple phone lines", examples=["Yes", "No", "No phone service"]
    )

    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Type of internet service used by the customer", examples=["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online security service", examples=["Yes", "No", "No internet service"]
    )

    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online backup service", examples=["Yes", "No", "No internet service"]
    )

    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has device protection service", examples=["Yes", "No", "No internet service"]
    )

    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has technical support service", examples=["Yes", "No", "No internet service"]
    )

    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer uses streaming TV service",  examples=["Yes", "No", "No internet service"]
    )

    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer uses streaming movies service", examples=["Yes", "No", "No internet service"]
    )

    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract type of the customer", examples=["Month-to-month", "One year", "Two year"]
    )

    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer uses paperless billing", examples=["Yes", "No"]
    )

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Payment method used by the customer", examples=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    MonthlyCharges: float = Field(
        ..., ge=0, description="Monthly amount charged to the customer", examples=[29.85, 56.95, 53.85, 42.30, 70.70]
    )

    TotalCharges: float = Field(
        ..., ge=0, description="Total amount charged to the customer", examples=[29.85, 56.95, 53.85, 42.30, 70.70]
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Churn prediction", examples=['Yes', 'No'])
    probability: Optional[float] = Field(None, description="Churn probability", examples=[0.5, 0.8])
    latency: float = Field(..., description="Latency of the prediction", examples=[0.5, 0.8])