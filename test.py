import pandas as pd
import numpy as np

# Set seed for reproducible hardware telemetry
np.random.seed(42)
num_rows = 50000

# 1. Identifiers & Driving Context
frame_ids = [f"FRM_{i:06d}" for i in range(1, num_rows + 1)]
speed_kmh = np.round(np.random.uniform(0.0, 130.0, size=num_rows), 1)

# 2. 18 Hardware, RF, & Environmental Features
radar_freq_ghz = np.round(np.random.normal(77.5, 0.2, size=num_rows), 3)
tx_power_dbm = np.round(np.random.normal(12.0, 0.5, size=num_rows), 2)
rx_gain_db = np.round(np.random.normal(35.0, 2.0, size=num_rows), 1)

target_range_m = np.round(np.random.uniform(1.0, 200.0, size=num_rows), 2)
snr_db = np.round(30.0 - (target_range_m * 0.1) + np.random.normal(0, 2.0, size=num_rows), 2)
azimuth_deg = np.round(np.random.uniform(-45.0, 45.0, size=num_rows), 2)
elevation_deg = np.round(np.random.uniform(-10.0, 10.0, size=num_rows), 2)
doppler_mps = np.round(np.random.normal(0.0, 15.0, size=num_rows), 2)

lidar_density = np.random.randint(50, 2000, size=num_rows)
radar_rcs = np.round(np.random.normal(10.0, 5.0, size=num_rows), 2)

sensor_temp = np.round(np.random.normal(45.0, 8.0, size=num_rows), 2)
mcu_voltage = np.round(np.random.normal(3.30, 0.04, size=num_rows), 3)
can_load_pct = np.round(np.random.uniform(20.0, 85.0, size=num_rows), 1)
latency_ms = np.round(np.random.normal(15.0, 3.0, size=num_rows), 2)

ambient_lux = np.round(np.random.uniform(0.0, 100000.0, size=num_rows), 0)
precipitation_mmh = np.round(np.random.exponential(scale=1.5, size=num_rows), 2)
calibration_drift = np.round(np.random.uniform(0.0, 1.0, size=num_rows), 3)

# 3. Assemble Dataframe
df = pd.DataFrame({
    'frame_id': frame_ids,
    'vehicle_speed_kmh': speed_kmh,
    'radar_frequency_ghz': radar_freq_ghz,
    'tx_power_dbm': tx_power_dbm,
    'rx_gain_db': rx_gain_db,
    'snr_db': snr_db,
    'azimuth_angle_deg': azimuth_deg,
    'elevation_angle_deg': elevation_deg,
    'target_range_meters': target_range_m,
    'doppler_velocity_mps': doppler_mps,
    'lidar_point_density': lidar_density,
    'radar_rcs_dbm2': radar_rcs,
    'sensor_temp_c': sensor_temp,
    'mcu_voltage_v': mcu_voltage,
    'can_bus_load_pct': can_load_pct,
    'packet_latency_ms': latency_ms,
    'ambient_lux': ambient_lux,
    'precipitation_rate_mmh': precipitation_mmh,
    'calibration_drift_index': calibration_drift
})

# 4. Target Feature (20th Column) - Degradation/Fault Flag
degradation_score = (
    (df['snr_db'] < 8.0).astype(int) * 3.0 +
    (df['sensor_temp_c'] > 62.0).astype(int) * 2.5 +
    (df['calibration_drift_index'] > 0.75).astype(int) * 3.5 +
    (df['precipitation_rate_mmh'] > 8.0).astype(int) * 1.5 +
    np.random.normal(0, 0.5, size=num_rows)
)

df['sensor_degraded_flag'] = (degradation_score > 3.0).astype(int)

# Save file directly to your working directory
df.to_csv('autonomous_radar_telemetry_50k.csv', index=False)
print("Saved: autonomous_radar_telemetry_50k.csv (50,000 rows x 20 columns)")