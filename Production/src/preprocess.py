
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        
        self.general_health_order = [['Poor', 'Fair', 'Good', 'Very good', 'Excellent']]
        self.age_category_order = [['Young Adults (18-34)', 'Early Middle Age (35-44)', 
                                    'Late Middle Age (45-54)', 'Seniors (55-69)', 'Elderly (70+)']]
    
        self.binary_cols = [
            'PhysicalActivities', 
            'AlcoholDrinkers', 
            'HadDiabetes', 
            'DeafOrHardOfHearing', 
            'BlindOrVisionDifficulty', 
            'DifficultyConcentrating',    
            'DifficultyWalking', 
            'DifficultyDressingBathing',  
            'DifficultyErrands'          
        ]
        
        self.other_cols = ['Sex', 'RaceEthnicityCategory']

        self.health_encoder = OrdinalEncoder(categories=self.general_health_order, 
                                             handle_unknown='use_encoded_value', unknown_value=-1)
        
        self.age_encoder = OrdinalEncoder(categories=self.age_category_order, 
                                          handle_unknown='use_encoded_value', unknown_value=-1)
        
        self.other_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        self.binary_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}

    def fit(self, X, y=None):
        # Fit Health
        if 'GeneralHealth' in X.columns:
            self.health_encoder.fit(X[['GeneralHealth']])
            
        # Fit Age
        if 'AgeCategory' in X.columns:
            self.age_encoder.fit(X[['AgeCategory']])
            
        # Fit Others
        existing_cols = [c for c in self.other_cols if c in X.columns]
        if existing_cols:
            self.other_encoder.fit(X[existing_cols])
            
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        # 1. Transform Health
        if 'GeneralHealth' in X_transformed.columns:
            X_transformed['GeneralHealth'] = self.health_encoder.transform(X_transformed[['GeneralHealth']])
            
        # 2. Transform Age
        if 'AgeCategory' in X_transformed.columns:
            X_transformed['AgeCategory'] = self.age_encoder.transform(X_transformed[['AgeCategory']])
            
        # 3. Transform Others
        existing_cols = [c for c in self.other_cols if c in X_transformed.columns]
        if existing_cols:
            X_transformed[existing_cols] = self.other_encoder.transform(X_transformed[existing_cols])
            
        # 4. Transform Binary (Yes/No)
        for col in self.binary_cols:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].map(self.binary_map)
                X_transformed[col] = X_transformed[col].fillna(-1)
                
        return X_transformed