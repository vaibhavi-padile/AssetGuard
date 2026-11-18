import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

def train_maintenance_model(df: pd.DataFrame):
    """
    Trains an imbalance-aware gradient boosting classifier optimizing for early anomaly detection.
    """
    features = ['temperature', 'vibration_amplitude', 'temp_rolling_mean', 'vibe_rolling_std']
    X = df[features]
    y = df['failure_target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    balance_ratio = neg_count / max(1, pos_count)
    
    model = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=4,
        scale_pos_weight=balance_ratio,
        random_state=42,
        eval_metric="logloss"
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    metrics = {
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0)
    }
    
    return model, metrics, features
