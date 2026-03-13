import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_temp_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    return df


def add_power(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * 3.14159 / 60)
    return df


def add_wear_torque_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wear_torque_interaction"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features")
    df = add_temp_diff(df)
    df = add_power(df)
    df = add_wear_torque_interaction(df)
    logger.info("Added engineered features: temp_diff, power, wear_torque_interaction")
    return df
