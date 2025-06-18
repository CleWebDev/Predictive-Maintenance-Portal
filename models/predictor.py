"""
models/predictor.py
Machine Learning model for predicting equipment failure
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class MaintenancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained_flag = False
        
    def _encode_categorical_features(self, df, fit_encoders=False):
        """Encode categorical features"""
        categorical_cols = ['machine_type', 'operating_mode', 'environment']
        
        for col in categorical_cols:
            if fit_encoders:
                # Create and fit encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    known_categories = set(le.classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_categories else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        
        return df
    
    def train(self, X, y):
        """Train the prediction model"""
        try:
            # Make a copy to avoid modifying original data
            X_encoded = X.copy()
            
            # Encode categorical features
            X_encoded = self._encode_categorical_features(X_encoded, fit_encoders=True)
            
            # Store feature names
            self.feature_names = X_encoded.columns.tolist()
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Validate model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model validation - MAE: {mae:.2f} days, R²: {r2:.3f}")
            
            self.is_trained_flag = True
            
            # Save model for production use
            self._save_model()
            
        except Exception as e:
            print(f"Error training model: {e}")
            raise
    
    def predict(self, input_data):
        """Make prediction for single input"""
        if not self.is_trained_flag:
            raise ValueError("Model not trained yet")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            
            # Encode categorical features
            df = self._encode_categorical_features(df, fit_encoders=False)
            
            # Ensure columns are in correct order
            df = df[self.feature_names]
            
            # Make prediction
            raw_prediction = self.model.predict(df)[0]
            
            # BUSINESS LOGIC FIXES:
            # 1. Cap prediction at 90% of maintenance interval
            maintenance_interval = input_data['maintenance_interval']
            max_allowed_days = maintenance_interval * 0.9
            
            # 2. Ensure prediction is reasonable (between 1 and max_allowed_days)
            prediction = max(1, min(max_allowed_days, raw_prediction))
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if not self.is_trained_flag:
            return {}
        
        importance_dict = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = round(importance, 3)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def explain_prediction(self, input_data, feature_importance):
        """Generate explanation for prediction"""
        explanations = []
        
        # High impact factors
        top_features = list(feature_importance.keys())[:3]
        
        for feature in top_features:
            if feature in input_data:
                value = input_data[feature]
                impact = feature_importance[feature]
                
                if feature == 'vibration_level_mm_s':
                    if value > 2.5:
                        explanations.append(f"High vibration level ({value:.1f} mm/s) increases failure risk")
                    elif value < 1.0:
                        explanations.append(f"Low vibration level ({value:.1f} mm/s) indicates good condition")
                
                elif feature == 'temperature_F':
                    if value > 85:
                        explanations.append(f"High temperature ({value:.1f}°F) accelerates wear")
                    elif value < 60:
                        explanations.append(f"Optimal temperature ({value:.1f}°F) extends equipment life")
                
                elif feature == 'load_factor':
                    if value > 0.8:
                        explanations.append(f"High load factor ({value:.1f}) increases stress on components")
                    elif value < 0.3:
                        explanations.append(f"Low load factor ({value:.1f}) reduces equipment stress")
                
                elif feature == 'maintenance_interval':
                    if value > 150:
                        explanations.append(f"Long maintenance interval ({value} days) may increase failure risk")
                    elif value < 60:
                        explanations.append(f"Frequent maintenance ({value} days) helps prevent failures")
        
        return explanations if explanations else ["Prediction based on overall equipment condition"]
    
    def _save_model(self):
        """Save trained model"""
        try:
            os.makedirs('models/saved', exist_ok=True)
            joblib.dump(self.model, 'models/saved/maintenance_model.pkl')
            joblib.dump(self.label_encoders, 'models/saved/label_encoders.pkl')
            joblib.dump(self.feature_names, 'models/saved/feature_names.pkl')
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    def load_model(self):
        """Load saved model"""
        try:
            self.model = joblib.load('models/saved/maintenance_model.pkl')
            self.label_encoders = joblib.load('models/saved/label_encoders.pkl')
            self.feature_names = joblib.load('models/saved/feature_names.pkl')
            self.is_trained_flag = True
            return True
        except Exception as e:
            print(f"Could not load saved model: {e}")
            return False
    
    def is_trained(self):
        """Check if model is trained"""
        return self.is_trained_flag