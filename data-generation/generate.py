import numpy as np
import pandas as pd

# Configuration
n_samples = 1000
np.random.seed(42)

# Categorical features and their options
machine_types = ['CNC Mill', 'Lathe', 'Compressor', 'Pump', 'Motor', 'Conveyor']
maintenance_intervals = [30, 60, 90, 120, 180]  # days
operating_modes = ['Idle', 'Standby', 'Production', 'Overload']
environments = ['Indoor', 'Outdoor', 'Dusty', 'Humid']

# Generate categorical data
machine_type = np.random.choice(machine_types, n_samples)
maintenance_interval = np.random.choice(maintenance_intervals, n_samples)
operating_mode = np.random.choice(operating_modes, n_samples)
environment = np.random.choice(environments, n_samples)

# Generate numeric sensor readings
vibration_level = np.random.normal(loc=2.0, scale=0.5, size=n_samples)  # mm/s
temperature = np.random.normal(loc=75, scale=10, size=n_samples)         # °F
pressure = np.random.normal(loc=100, scale=15, size=n_samples)           # PSI
load_factor = np.random.uniform(0, 1, size=n_samples)                    # 0 to 1

# Synthetic target: days-to-failure (regression)
# Assumed relationship: higher vibration, temperature, load shorten days-to-failure
noise = np.random.normal(0, 5, n_samples)
days_to_failure = (
    200 
    - 20 * vibration_level 
    - 0.5 * (temperature - 70) 
    - 50 * load_factor
    + noise
)
days_to_failure = np.clip(days_to_failure, a_min=1, a_max=None)

# Assemble DataFrame
df = pd.DataFrame({
    'machine_type': machine_type,
    'maintenance_interval': maintenance_interval,
    'operating_mode': operating_mode,
    'environment': environment,
    'vibration_level_mm_s': vibration_level,
    'temperature_F': temperature,
    'pressure_PSI': pressure,
    'load_factor': load_factor,
    'days_to_failure': days_to_failure
})

# Save to CSV
df.to_csv('synthetic_maintenance_data.csv', index=False)

# Display first few rows
df.head()
