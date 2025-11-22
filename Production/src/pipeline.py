
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

try:
    from src.preprocess import CustomPreprocessor
except ModuleNotFoundError:
    from preprocess import CustomPreprocessor
# -------------------

def build_pipeline():
    """
    Creates ML pipeline:
        Preprocess  ->  CatBoost Model
    """
    pipeline = Pipeline([
        ("preprocess", CustomPreprocessor()),
        ("model", CatBoostClassifier(
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            loss_function='Logloss',
            eval_metric='F1',
            verbose=200,
            random_state=42,
            allow_writing_files=False
        ))
    ])
    return pipeline