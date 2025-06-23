"""
utils/data_loader.py
Utility for loading and preparing maintenance data
"""

import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, csv_path='data/realistic_maintenance_data.csv'):
        self.csv_path = csv_path
        self.df = None
        self.unique_values = {}
        
    def load_and_prepare_data(self):
        """Load CSV data and prepare for ML model"""
        try:
            # Load data
            self.df = pd.read_csv(self.csv_path)
            
            print(f"Loaded {len(self.df)} records from {self.csv_path}")
            
            # Basic data validation
            self._validate_data()
            
            # Extract unique values for UI dropdowns
            self._extract_unique_values()
            
            # Prepare features (X) and target (y)
            feature_columns = [
                'machine_type', 'maintenance_interval', 'operating_mode', 
                'environment', 'vibration_level_mm_s', 'temperature_F', 
                'pressure_PSI', 'load_factor'
            ]
            
            X = self.df[feature_columns].copy()
            y = self.df['days_to_failure'].copy()
            
            # Handle any missing values
            X = self._handle_missing_values(X)
            
            print(f"Prepared {len(X)} samples with {len(X.columns)} features")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _validate_data(self):
        """Basic data validation"""
        # Check for required columns
        required_cols = [
            'machine_type', 'maintenance_interval', 'operating_mode',
            'environment', 'vibration_level_mm_s', 'temperature_F',
            'pressure_PSI', 'load_factor', 'days_to_failure'
        ]
        
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for reasonable value ranges
        if self.df['days_to_failure'].min() < 0:
            print("Warning: Negative days_to_failure values found")
        
        if self.df['temperature_F'].min() < -100 or self.df['temperature_F'].max() > 200:
            print("Warning: Temperature values outside expected range")
        
        if self.df['load_factor'].min() < 0 or self.df['load_factor'].max() > 1:
            print("Warning: Load factor values outside 0-1 range")
        
        print("Data validation completed")
    
    def _extract_unique_values(self):
        """Extract unique values for categorical columns"""
        categorical_columns = ['machine_type', 'operating_mode', 'environment']
        
        for col in categorical_columns:
            self.unique_values[col] = sorted(self.df[col].unique().tolist())
        
        # Add numerical ranges for reference
        self.unique_values['maintenance_interval_range'] = {
            'min': int(self.df['maintenance_interval'].min()),
            'max': int(self.df['maintenance_interval'].max())
        }
        
        # Create dropdown options for vibration level (mm/s)
        vibration_min = self.df['vibration_level_mm_s'].min()
        vibration_max = self.df['vibration_level_mm_s'].max()
        self.unique_values['vibration_level_options'] = self._create_numerical_options(
            vibration_min, vibration_max, step=0.2, decimal_places=1
        )
        
        # Create dropdown options for pressure (PSI)
        pressure_min = self.df['pressure_PSI'].min()
        pressure_max = self.df['pressure_PSI'].max()
        self.unique_values['pressure_options'] = self._create_numerical_options(
            pressure_min, pressure_max, step=5, decimal_places=0
        )
        
        # Keep ranges for temperature and load factor (still text inputs)
        numerical_cols = ['temperature_F', 'load_factor']
        for col in numerical_cols:
            self.unique_values[f'{col}_range'] = {
                'min': round(self.df[col].min(), 2),
                'max': round(self.df[col].max(), 2),
                'avg': round(self.df[col].mean(), 2)
            }
    
    def _create_numerical_options(self, min_val, max_val, step, decimal_places=1):
        """Create a list of numerical options for dropdown"""
        options = []
        current = min_val
        
        # Round min to nearest step for cleaner options
        current = round(current / step) * step
        
        while current <= max_val:
            if decimal_places == 0:
                options.append(int(current))
            else:
                options.append(round(current, decimal_places))
            current += step
        
        # Ensure we include the max value if it wasn't included
        if decimal_places == 0:
            max_rounded = int(max_val)
        else:
            max_rounded = round(max_val, decimal_places)
            
        if max_rounded not in options:
            options.append(max_rounded)
        
        return sorted(list(set(options)))  # Remove duplicates and sort
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill numerical missing values with median
        numerical_cols = ['maintenance_interval', 'vibration_level_mm_s', 
                         'temperature_F', 'pressure_PSI', 'load_factor']
        
        for col in numerical_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"Filled {col} missing values with median: {median_value}")
        
        # Fill categorical missing values with mode
        categorical_cols = ['machine_type', 'operating_mode', 'environment']
        
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"Filled {col} missing values with mode: {mode_value}")
        
        return df
    
    def get_unique_values(self):
        """Return unique values for UI components"""
        if not self.unique_values:
            self.load_and_prepare_data()
        return self.unique_values
    
    def get_sample_data(self, n_samples=5):
        """Get sample data for testing"""
        if self.df is None:
            self.load_and_prepare_data()
        
        return self.df.head(n_samples).to_dict('records')
    
    def get_data_statistics(self):
        """Get basic statistics about the dataset"""
        if self.df is None:
            self.load_and_prepare_data()
        
        stats = {
            'total_records': len(self.df),
            'machine_type_distribution': self.df['machine_type'].value_counts().to_dict(),
            'average_days_to_failure': round(self.df['days_to_failure'].mean(), 1),
            'min_days_to_failure': round(self.df['days_to_failure'].min(), 1),
            'max_days_to_failure': round(self.df['days_to_failure'].max(), 1)
        }
        
        return stats