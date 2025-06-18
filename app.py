#!/usr/bin/env python3
"""
Predictive Maintenance & Service Request Portal
Main Flask application entry point
"""

import os
from flask import Flask, render_template, request, jsonify
from models.predictor import MaintenancePredictor
from utils.data_loader import DataLoader
from utils.risk_calculator import RiskCalculator

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # Initialize components
    data_loader = DataLoader('data/synthetic_maintenance_data.csv')
    predictor = MaintenancePredictor()
    risk_calculator = RiskCalculator()
    
    # Train model on startup
    print("Loading data and training model...")
    X, y = data_loader.load_and_prepare_data()
    predictor.train(X, y)
    print("Model training completed!")
    
    @app.route('/')
    def index():
        """Main portal page"""
        # Get unique values for dropdowns
        unique_values = data_loader.get_unique_values()
        return render_template('index.html', unique_values=unique_values)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """API endpoint for predictions"""
        try:
            # Parse input data
            input_data = {
                'machine_type': request.json.get('machine_type'),
                'maintenance_interval': int(request.json.get('maintenance_interval')),
                'operating_mode': request.json.get('operating_mode'),
                'environment': request.json.get('environment'),
                'vibration_level_mm_s': float(request.json.get('vibration_level_mm_s')),
                'temperature_F': float(request.json.get('temperature_F')),
                'pressure_PSI': float(request.json.get('pressure_PSI')),
                'load_factor': float(request.json.get('load_factor'))
            }
            
            # Make prediction
            days_until_maintenance = predictor.predict(input_data)
            
            # Calculate risk assessment
            risk_info = risk_calculator.calculate_risk(days_until_maintenance)
            
            # Get feature importance for explanation
            feature_importance = predictor.get_feature_importance()
            
            response = {
                'days_until_maintenance': round(days_until_maintenance, 1),
                'maintenance_interval': input_data['maintenance_interval'],
                'risk_level': risk_info['level'],
                'risk_color': risk_info['color'],
                'risk_message': risk_info['message'],
                'should_schedule_service': risk_info['should_schedule'],
                'feature_importance': feature_importance,
                'explanation': predictor.explain_prediction(input_data, feature_importance)
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/schedule_service', methods=['POST'])
    def schedule_service():
        """Handle service scheduling requests"""
        try:
            data = request.json
            machine_type = data.get('machine_type')
            predicted_days = data.get('days_until_maintenance')
            
            # In a real app, this would integrate with a scheduling system
            # For now, we'll just return a confirmation
            
            response = {
                'success': True,
                'message': f'Maintenance request submitted for {machine_type}. '
                          f'Recommended within {predicted_days} days. '
                          f'A service technician will contact you within 24 hours.',
                'ticket_id': f'MNT-{hash(str(data)) % 10000:04d}'
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/health')
    def health_check():
        """Health check endpoint for deployment"""
        return jsonify({'status': 'healthy', 'model_loaded': predictor.is_trained()})
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') == 'development')