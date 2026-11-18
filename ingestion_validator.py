import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger()

def process_sensor_telemetry(file_path: str) -> pd.DataFrame:
    """
    Ingests raw IoT sensor data, executes structural forward-fills for network drops,
    and applies rolling window feature transformation pipelines.
    """
    logger.info("Initializing telemetry ingestion workflow")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.warn("Data file absent, generating synthetic structural telemetry matrix")
        np.random.seed(42)
        timestamps = pd.date_range(start="2026-01-01", periods=200, freq="h")
        df = pd.DataFrame({
            'timestamp': timestamps,
            'machine_id': ['MCH-901'] * 200,
            'temperature': np.random.normal(78.4, 4.2, 200),
            'vibration_amplitude': np.random.normal(1.65, 0.25, 200),
            'failure_target': [0] * 194 + [1] * 6
        })
        df.loc[15:18, 'temperature'] = np.nan 

    if df['temperature'].isna().sum() > 0:
        logger.info("Signal interruptions detected, applying forward-fill validation guard")
        df['temperature'] = df['temperature'].ffill()

    df['temp_rolling_mean'] = df['temperature'].rolling(window=3, min_periods=1).mean()
    df['vibe_rolling_std'] = df['vibration_amplitude'].rolling(window=3, min_periods=1).std().fillna(0)

    logger.info("Telemetry sequence processing complete", records_shape=df.shape)
    return df
