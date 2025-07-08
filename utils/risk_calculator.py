"""
utils/risk_calculator.py
Risk assessment logic for maintenance predictions
"""

class RiskCalculator:
    def __init__(self):
        # More dramatic risk thresholds for better differentiation
        self.risk_thresholds = {
            'critical': 10,   # < 10 days - immediate action
            'high': 25,       # 10-25 days - urgent
            'medium': 45,     # 25-45 days - soon
            'low': 70         # 45-70 days - monitor
            # > 70 days = very low risk
        }
        
        self.risk_colors = {
            'critical': 'rgba(255, 59, 48, 0.9)',   # Apple Red
            'high': 'rgba(255, 149, 0, 0.9)',       # Apple Orange
            'medium': 'rgba(255, 204, 0, 0.9)',     # Apple Yellow
            'low': 'rgba(52, 199, 89, 0.9)',        # Apple Green
            'very_low': 'rgba(0, 122, 255, 0.9)'    # Apple Blue
        }
    
    def calculate_risk(self, days_until_maintenance_needed):
        """Calculate risk level based on predicted days until maintenance needed"""
        days = float(days_until_maintenance_needed)
        
        if days < self.risk_thresholds['critical']:
            level = 'critical'
            message = f"CRITICAL: Maintenance needed within {days:.1f} days. Schedule immediately!"
            should_schedule = True
            
        elif days < self.risk_thresholds['high']:
            level = 'high'
            message = f"HIGH PRIORITY: Maintenance recommended within {days:.1f} days. Schedule soon."
            should_schedule = True
            
        elif days < self.risk_thresholds['medium']:
            level = 'medium'
            message = f"MEDIUM PRIORITY: Maintenance should be scheduled within {days:.1f} days."
            should_schedule = True
            
        elif days < self.risk_thresholds['low']:
            level = 'low'
            message = f"LOW PRIORITY: Maintenance can wait {days:.1f} days. Monitor condition."
            should_schedule = False
            
        else:
            level = 'very_low'
            message = f"VERY LOW PRIORITY: Equipment in good condition. Next maintenance in {days:.1f} days."
            should_schedule = False
        
        return {
            'level': level,
            'color': self.risk_colors[level],
            'message': message,
            'should_schedule': should_schedule,
            'days_to_maintenance': days,
            'priority': self._get_priority(level)
        }
    
    def _get_priority(self, risk_level):
        """Get numerical priority for sorting (1 = highest priority)"""
        priority_map = {
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4,
            'very_low': 5
        }
        return priority_map.get(risk_level, 5)
    
    def get_maintenance_recommendations(self, risk_info, machine_type):
        """Get specific maintenance recommendations based on risk and machine type"""
        recommendations = []
        
        risk_level = risk_info['level']
        days = risk_info['days_to_failure']
        
        # General recommendations based on risk level
        if risk_level == 'critical':
            recommendations.extend([
                "Stop equipment operation immediately",
                "Contact emergency service technician",
                "Inspect for signs of imminent failure",
                "Prepare replacement parts if available"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "Schedule service within 1-2 days",
                "Increase monitoring frequency",
                "Reduce operating load if possible",
                "Prepare for potential downtime"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "Schedule preventive maintenance within 1-2 weeks",
                "Monitor key parameters daily",
                "Review maintenance history"
            ])
        else:
            recommendations.extend([
                "Continue normal operations",
                "Follow standard monitoring schedule",
                "Consider next scheduled maintenance"
            ])
        
        # Machine-specific recommendations
        machine_specific = self._get_machine_specific_recommendations(machine_type, risk_level)
        recommendations.extend(machine_specific)
        
        return recommendations
    
    def _get_machine_specific_recommendations(self, machine_type, risk_level):
        """Get machine-type specific recommendations"""
        recommendations = []
        
        if machine_type.lower() == 'pump':
            if risk_level in ['critical', 'high']:
                recommendations.extend([
                    "Check for cavitation or impeller damage",
                    "Inspect seals and gaskets",
                    "Monitor for unusual vibration or noise"
                ])
            else:
                recommendations.append("Check fluid levels and filters")
                
        elif machine_type.lower() == 'motor':
            if risk_level in ['critical', 'high']:
                recommendations.extend([
                    "Check motor temperature and ventilation",
                    "Inspect electrical connections",
                    "Listen for bearing noise"
                ])
            else:
                recommendations.append("Standard electrical safety checks")
                
        elif machine_type.lower() == 'compressor':
            if risk_level in ['critical', 'high']:
                recommendations.extend([
                    "Check pressure relief valves",
                    "Inspect air intake filters",
                    "Monitor compressor oil levels"
                ])
            else:
                recommendations.append("Regular filter and oil maintenance")
                
        elif machine_type.lower() in ['lathe', 'cnc mill']:
            if risk_level in ['critical', 'high']:
                recommendations.extend([
                    "Check tool wear and alignment",
                    "Inspect spindle bearings",
                    "Verify coolant system operation"
                ])
            else:
                recommendations.append("Regular lubrication and calibration")
                
        elif machine_type.lower() == 'conveyor':
            if risk_level in ['critical', 'high']:
                recommendations.extend([
                    "Inspect belt tension and alignment",
                    "Check roller bearings",
                    "Look for signs of belt wear"
                ])
            else:
                recommendations.append("Standard belt and roller inspection")
        
        return recommendations
    
    def calculate_maintenance_cost_impact(self, risk_info):
        """Estimate cost impact of different maintenance approaches"""
        days = risk_info['days_to_failure']
        risk_level = risk_info['level']
        
        # Base maintenance costs (example values)
        preventive_cost = 1000
        reactive_cost = 5000  # Typically 5x more expensive
        downtime_cost_per_hour = 500
        
        if risk_level == 'critical':
            estimated_downtime = 24  # hours
            emergency_multiplier = 2.0
            total_cost = (reactive_cost * emergency_multiplier) + (downtime_cost_per_hour * estimated_downtime)
            
        elif risk_level == 'high':
            estimated_downtime = 8
            total_cost = reactive_cost + (downtime_cost_per_hour * estimated_downtime)
            
        else:
            estimated_downtime = 2  # Planned maintenance
            total_cost = preventive_cost + (downtime_cost_per_hour * estimated_downtime)
        
        savings = (reactive_cost + (downtime_cost_per_hour * 24)) - total_cost
        
        return {
            'estimated_cost': total_cost,
            'estimated_downtime_hours': estimated_downtime,
            'potential_savings': max(0, savings),
            'cost_category': 'Emergency' if risk_level in ['critical', 'high'] else 'Planned'
        }