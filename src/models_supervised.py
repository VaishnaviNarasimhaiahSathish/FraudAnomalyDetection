from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def build_log_reg(class_weights):
    """Logistic Regression with class weights."""
    return LogisticRegression(class_weight=class_weights, max_iter=500)


def build_rf(class_weights):
    """Random Forest with class weights."""
    return RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weights,
        random_state=42
    )


def build_xgb(class_weights):
    """XGBoost model exactly as defined in notebook."""
    return xgb.XGBClassifier(
        scale_pos_weight=class_weights[1],
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
