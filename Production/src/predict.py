import joblib
import pandas as pd
import os

# Import Fix: عشان يشتغل سواء من جوه الفولدر أو من بره
try:
    from src.preprocess import CustomPreprocessor
except ModuleNotFoundError:
    from preprocess import CustomPreprocessor

class Predictor:
    def __init__(self, model_path="model/pipeline.pkl"):
        self.model_path = model_path
        self.model = None
        self.expected_columns = [] 
        self.load_model()

    def load_model(self):
        # 1. تصحيح مسار الموديل لو مش موجود
        if not os.path.exists(self.model_path):
            if os.path.exists(f"../{self.model_path}"):
                self.model_path = f"../{self.model_path}"
            else:
                raise FileNotFoundError(f"❌ Model file not found at: {self.model_path}")

        print(f"⏳ Loading model from: {self.model_path}...")
        self.model = joblib.load(self.model_path)
        
        # ---------------------------------------------------------
        # 🔥 الحل السحري: استخراج ترتيب الأعمدة من الموديل نفسه
        # ---------------------------------------------------------
        try:
            # بنحاول نجيب أسماء الأعمدة من الـ Pipeline مباشرة
            if hasattr(self.model, 'feature_names_in_'):
                self.expected_columns = list(self.model.feature_names_in_)
            
            # لو الموديل عبارة عن Pipeline، ندخل جوه آخر خطوة (CatBoost)
            elif hasattr(self.model.steps[-1][1], 'feature_names_in_'):
                self.expected_columns = list(self.model.steps[-1][1].feature_names_in_)
            
            else:
                 print("⚠️ Warning: Could not detect feature names automatically.")
                 
        except Exception as e:
            print(f"⚠️ Warning during feature detection: {e}")

        print(f"✅ Model loaded! Expecting features: {self.expected_columns}")


    def predict(self, input_data: dict):
        """
        بياخد البيانات كـ Dictionary ويرجع النتيجة
        """
        df = pd.DataFrame([input_data])

        # 1. لو الموديل قدر يعرف الأسماء، نستخدمها للترتيب
        if self.expected_columns:
            # التأكد من وجود كل الأعمدة المطلوبة
            missing = [c for c in self.expected_columns if c not in df.columns]
            if missing:
                raise ValueError(f"❌ Missing columns: {missing}")
            
            # --- أهم سطر: إعادة الترتيب عشان يطابق الموديل ---
            df = df[self.expected_columns]
        
        # 2. التوقع
        prediction = self.model.predict(df)
        probability = self.model.predict_proba(df)

        result_class = int(prediction[0])
        prob_score = probability[0][1]

        return {
            "prediction": result_class,
            "probability": prob_score,
            "label": "High Risk ⚠️" if result_class == 1 else "Low Risk ✅"
        }

# ---------------------------------------------------------
# Main Execution (للتجربة)
# ---------------------------------------------------------
if __name__ == "__main__":
    # عينة تجربة
    sample_data = {
        'Sex': 'Male',
        'GeneralHealth': 'Good',
        'PhysicalHealthDays': 0.0,
        'MentalHealthDays': 0.0,
        'PhysicalActivities': 'Yes',
        'SleepHours': 7.0,
        'HadDiabetes': 'No',
        'DeafOrHardOfHearing': 'No',
        'BlindOrVisionDifficulty': 'No',
        'DifficultyConcentrating': 'No',
        'DifficultyWalking': 'No',
        'DifficultyDressingBathing': 'No',
        'DifficultyErrands': 'No',
        'RaceEthnicityCategory': 'White only, Non-Hispanic',
        'AgeCategory': 'Young Adults (18-34)',
        'HeightInMeters': 1.75,
        'WeightInKilograms': 80.0,
        'AlcoholDrinkers': 'No'
    }

    try:
        predictor = Predictor(model_path="model/pipeline.pkl")
        result = predictor.predict(sample_data)
        print(f"Result: {result['label']}")
    except Exception as e:
        print(f"Error: {e}")
        