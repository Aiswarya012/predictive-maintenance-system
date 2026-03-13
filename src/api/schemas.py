from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    air_temperature_k: float = Field(..., description="Air temperature in Kelvin")
    process_temperature_k: float = Field(..., description="Process temperature in Kelvin")
    rotational_speed_rpm: float = Field(..., description="Rotational speed in RPM")
    torque_nm: float = Field(..., description="Torque in Nm")
    tool_wear_min: float = Field(..., description="Tool wear in minutes")
    machine_type: str = Field(..., description="Machine quality type: L, M, or H")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "air_temperature_k": 298.1,
                    "process_temperature_k": 308.6,
                    "rotational_speed_rpm": 1551,
                    "torque_nm": 42.8,
                    "tool_wear_min": 0,
                    "machine_type": "M",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: str
    failure_probability: float
    status: str = "success"
