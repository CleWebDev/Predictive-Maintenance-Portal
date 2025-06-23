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
        
    def _engineer_features(self, df):
        """Create more sophisticated features that better reflect real-world relationships"""
        df = df.copy()
        
        # Vibration risk score (exponential relationship)
        df['vibration_risk'] = df['vibration_level_mm_s'] ** 2.5
        
        # Temperature stress factor (higher temperatures accelerate wear exponentially)
        df['temp_stress'] = np.where(df['temperature_F'] > 80, 
                                   (df['temperature_F'] - 60) ** 2 / 100,
                                   1.0)
        
        # Pressure strain (extreme pressures are much worse)
        df['pressure_strain'] = np.where(df['pressure_PSI'] > 120,
                                       (df['pressure_PSI'] - 80) ** 1.8 / 50,
                                       np.where(df['pressure_PSI'] < 70,
                                              (80 - df['pressure_PSI']) ** 1.5 / 30,
                                              1.0))
        
        # Load fatigue (high load factors cause exponential wear)
        df['load_fatigue'] = df['load_factor'] ** 3
        
        # Combined stress index (multiplicative effects)
        df['combined_stress'] = (df['vibration_risk'] * df['temp_stress'] * 
                                df['pressure_strain'] * (1 + df['load_fatigue']))
        
        # Operating severity (some modes are much worse)
        mode_severity = {
            'Idle': 0.3,
            'Standby': 0.5, 
            'Production': 1.0,
            'Overload': 3.5  # Overload is dramatically worse
        }
        df['operating_severity'] = df['operating_mode'].map(mode_severity).fillna(1.0)
        
        # Environment impact
        env_impact = {
            'Indoor': 0.7,
            'Outdoor': 1.2,
            'Humid': 1.8,
            'Dusty': 2.5  # Dusty environments are much worse
        }
        df['env_impact'] = df['environment'].map(env_impact).fillna(1.0)
        
        # Machine-specific vulnerability
        machine_vulnerability = {
            'Pump': 1.0,
            'Motor': 0.8,
            'Compressor': 1.4,
            'Lathe': 1.1,
            'Conveyor': 0.9,
            'CNC Mill': 1.3
        }
        df['machine_vulnerability'] = df['machine_type'].map(machine_vulnerability).fillna(1.0)
        
        # Maintenance interval pressure (longer intervals = higher risk)
        df['maintenance_pressure'] = (df['maintenance_interval'] / 90) ** 1.5
        
        # Final degradation rate (this drives the prediction)
        df['degradation_rate'] = (df['combined_stress'] * df['operating_severity'] * 
                                 df['env_impact'] * df['machine_vulnerability'] * 
                                 df['maintenance_pressure'])
        
        return df
        
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
        """Train the prediction model with enhanced feature engineering"""
        try:
            # Make a copy to avoid modifying original data
            X_enhanced = X.copy()
            
            # Add sophisticated feature engineering
            X_enhanced = self._engineer_features(X_enhanced)
            
            # Encode categorical features
            X_enhanced = self._encode_categorical_features(X_enhanced, fit_encoders=True)
            
            # Store feature names
            self.feature_names = X_enhanced.columns.tolist()
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_enhanced, y, test_size=0.2, random_state=42
            )
            
            # Train model with more sophisticated parameters
            self.model = RandomForestRegressor(
                n_estimators=200,  # More trees
                random_state=42,
                max_depth=15,      # Deeper trees
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='sqrt'
            )
            
            self.model.fit(X_train, y_train)
            
            # Validate model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Enhanced model validation - MAE: {mae:.2f} days, R²: {r2:.3f}")
            
            self.is_trained_flag = True
            
            # Save model for production use
            self._save_model()
            
        except Exception as e:
            print(f"Error training model: {e}")
            raise
    
    def predict(self, input_data):
        """Make prediction for single input with realistic business logic"""
        if not self.is_trained_flag:
            raise ValueError("Model not trained yet")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            
            # Apply feature engineering
            df = self._engineer_features(df)
            
            # Encode categorical features
            df = self._encode_categorical_features(df, fit_encoders=False)
            
            # Ensure columns are in correct order
            df = df[self.feature_names]
            
            # Make prediction
            raw_prediction = self.model.predict(df)[0]
            
            # REALISTIC BUSINESS LOGIC:
            maintenance_interval = input_data['maintenance_interval']
            
            # Get the degradation rate for more intelligent capping
            degradation_rate = df['degradation_rate'].iloc[0]
            
            # Dynamic capping based on conditions
            if degradation_rate > 10:  # Extreme conditions
                max_allowed_days = maintenance_interval * 0.3  # Emergency maintenance
            elif degradation_rate > 5:  # High stress
                max_allowed_days = maintenance_interval * 0.5  # Urgent maintenance  
            elif degradation_rate > 2:  # Moderate stress
                max_allowed_days = maintenance_interval * 0.7  # Early maintenance
            else:  # Good conditions
                max_allowed_days = maintenance_interval * 0.95  # Normal schedule
            
            # Ensure prediction is reasonable (between 5 and max_allowed_days)
            prediction = max(5, min(max_allowed_days, raw_prediction))
            
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
        """Generate detailed, realistic explanations for prediction"""
        explanations = []
        
        # Calculate key metrics for explanation
        vibration = input_data['vibration_level_mm_s']
        temperature = input_data['temperature_F']
        pressure = input_data['pressure_PSI']
        load_factor = input_data['load_factor']
        operating_mode = input_data['operating_mode']
        environment = input_data['environment']
        
        # Vibration analysis
        if vibration > 3.0:
            explanations.append(f"CRITICAL: Excessive vibration ({vibration:.1f} mm/s) indicates severe mechanical issues - immediate inspection required")
        elif vibration > 2.5:
            explanations.append(f"HIGH CONCERN: Elevated vibration ({vibration:.1f} mm/s) suggests bearing wear or misalignment")
        elif vibration > 2.0:
            explanations.append(f"MODERATE: Above-normal vibration ({vibration:.1f} mm/s) warrants monitoring")
        elif vibration < 1.0:
            explanations.append(f"EXCELLENT: Low vibration ({vibration:.1f} mm/s) indicates good mechanical condition")
        
        # Temperature analysis
        if temperature > 90:
            explanations.append(f"DANGER: High temperature ({temperature:.1f}°F) accelerates wear exponentially - cooling system check needed")
        elif temperature > 80:
            explanations.append(f"WARNING: Elevated temperature ({temperature:.1f}°F) reduces component lifespan significantly")
        elif temperature < 65:
            explanations.append(f"OPTIMAL: Good operating temperature ({temperature:.1f}°F) extends equipment life")
        
        # Pressure analysis
        if pressure > 140:
            explanations.append(f"EXTREME: Dangerously high pressure ({pressure} PSI) risks catastrophic failure")
        elif pressure > 120:
            explanations.append(f"HIGH STRESS: Excessive pressure ({pressure} PSI) accelerates seal and gasket wear")
        elif pressure < 70:
            explanations.append(f"LOW EFFICIENCY: Insufficient pressure ({pressure} PSI) may indicate system leaks")
        
        # Load factor analysis
        if load_factor > 0.85:
            explanations.append(f"OVERLOADED: Extreme load ({load_factor*100:.0f}%) causes rapid fatigue - reduce load immediately")
        elif load_factor > 0.7:
            explanations.append(f"HEAVY DUTY: High load factor ({load_factor*100:.0f}%) increases maintenance frequency needs")
        elif load_factor < 0.3:
            explanations.append(f"LIGHT DUTY: Low load factor ({load_factor*100:.0f}%) allows extended maintenance intervals")
        
        # Operating mode impact
        if operating_mode == 'Overload':
            explanations.append("CRITICAL FACTOR: Overload operation dramatically shortens equipment lifespan")
        elif operating_mode == 'Production':
            explanations.append("STANDARD: Production mode requires regular maintenance schedule")
        elif operating_mode == 'Idle':
            explanations.append("FAVORABLE: Idle operation reduces wear and extends service intervals")
        
        # Environmental impact
        if environment == 'Dusty':
            explanations.append("HARSH CONDITIONS: Dusty environment clogs filters and accelerates abrasive wear")
        elif environment == 'Humid':
            explanations.append("CORROSIVE: High humidity promotes rust and electrical component degradation")
        elif environment == 'Outdoor':
            explanations.append("VARIABLE: Outdoor conditions expose equipment to temperature cycling and contamination")
        elif environment == 'Indoor':
            explanations.append("PROTECTED: Indoor environment provides stable operating conditions")
        
        # Combined effect
        risk_factors = []
        if vibration > 2.5: risk_factors.append("high vibration")
        if temperature > 85: risk_factors.append("excessive heat")
        if pressure > 130: risk_factors.append("high pressure")
        if load_factor > 0.8: risk_factors.append("overloading")
        if operating_mode == 'Overload': risk_factors.append("overload mode")
        if environment in ['Dusty', 'Humid']: risk_factors.append("harsh environment")
        
        if len(risk_factors) >= 3:
            explanations.append(f"MULTIPLE STRESSORS: Combination of {', '.join(risk_factors)} creates compound failure risk")
        elif len(risk_factors) == 0:
            explanations.append("EXCELLENT CONDITIONS: All parameters within optimal ranges for extended operation")
        
        return explanations if explanations else ["Equipment condition assessment based on current sensor readings"]
    
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