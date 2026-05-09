# 02_Data_Preprocessing.py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("DATA PREPROCESSING & CLEANING WITH AQI CALCULATION")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n1. LOADING DATA...")
print("-" * 40)

# Load your dataset
df = pd.read_csv('D:\Intro to Data Science\global_air_quality_data_10000.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================
# 2. DOCUMENT DATA LIMITATIONS
# ============================================
print("\n2. DOCUMENTING DATA LIMITATIONS...")
print("-" * 40)

limitations_report = """
DATA LIMITATIONS DOCUMENTATION
================================

1. MISSING VALUES ANALYSIS:
"""
print(limitations_report)

# Check missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Values': missing_values,
    'Percentage': missing_percentage
})

print("Missing values in each column:")
print(missing_df[missing_df['Missing_Values'] > 0])

# Data quality issues
print("\n2. DATA QUALITY ISSUES:")
print("-" * 30)
print("a. Negative values in pollutants (if any):")
for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
    negative_count = (df[col] < 0).sum()
    if negative_count > 0:
        print(f"   {col}: {negative_count} negative values")

print("\nb. Unrealistic extreme values:")
print("   Temperature outside -50°C to 60°C range:", ((df['Temperature'] < -50) | (df['Temperature'] > 60)).sum())
print("   Humidity outside 0-100% range:", ((df['Humidity'] < 0) | (df['Humidity'] > 100)).sum())

print("\nc. Validating pollutant ranges:")
valid_ranges = {
    'PM2.5': (0, 500),    # µg/m³ - Delhi sometimes reaches 400-500
    'PM10': (0, 1000),    # µg/m³
    'NO2': (0, 400),      # µg/m³
    'SO2': (0, 500),      # µg/m³
    'CO': (0, 30),        # mg/m³
    'O3': (0, 400)        # µg/m³
}

for col, (low, high) in valid_ranges.items():
    invalid_count = ((df[col] < low) | (df[col] > high)).sum()
    if invalid_count > 0:
        print(f"   {col}: {invalid_count} values outside realistic range {low}-{high}")

print("\nd. Temporal coverage:")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"   Number of cities: {df['City'].nunique()}")
print(f"   Number of countries: {df['Country'].nunique()}")

# Save limitations to file
with open('data_limitations_report.txt', 'w') as f:
    f.write(limitations_report)
    f.write(str(missing_df))
print("\n✓ Data limitations saved to 'data_limitations_report.txt'")

# ============================================
# 3. DATA CLEANING
# ============================================
print("\n3. DATA CLEANING...")
print("-" * 40)

# Convert date
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day_of_Week'] = df['Date'].dt.dayofweek

# Handle unrealistic values before missing values
print("Handling unrealistic values...")
for col, (low, high) in valid_ranges.items():
    df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

# Handle missing values
print("Handling missing values...")
df_original = df.copy()
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    elif df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# Handle negative values (set to 0 for pollutants)
for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
    df[col] = df[col].clip(lower=0)

# ============================================
# 4. CORRECTED AQI CALCULATION (CORE REQUIREMENT)
# ============================================
print("\n4. CALCULATING AIR QUALITY INDEX (AQI)...")
print("-" * 40)

def calculate_aqi_component(concentration, breakpoints):
    """
    Calculate AQI for a single pollutant using EPA formula
    I_p = [(I_high - I_low)/(BP_high - BP_low)] × (C_p - BP_low) + I_low
    """
    for bp_low, bp_high, i_low, i_high in breakpoints:
        if bp_low <= concentration <= bp_high:
            aqi = ((i_high - i_low) / (bp_high - bp_low)) * (concentration - bp_low) + i_low
            return round(aqi)
    # If concentration exceeds highest breakpoint
    if concentration > breakpoints[-1][1]:
        return 500
    return 0

# EPA AQI breakpoints (24-hour averages) - CORRECT VALUES
# PM2.5 in µg/m³
pm25_breakpoints = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500)
]

# PM10 in µg/m³
pm10_breakpoints = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500)
]

# NO2 in ppb (convert from µg/m³: 1 ppb NO2 = 1.88 µg/m³ at 25°C)
# df['NO2_ppb'] = df['NO2'] / 1.88
no2_breakpoints = [
    (0, 53, 0, 50),      # 0-53 ppb
    (54, 100, 51, 100),  # 54-100 ppb
    (101, 360, 101, 150), # 101-360 ppb
    (361, 649, 151, 200), # 361-649 ppb
    (650, 1249, 201, 300), # 650-1249 ppb
    (1250, 1649, 301, 400), # 1250-1649 ppb
    (1650, 2049, 401, 500)  # 1650-2049 ppb
]

# SO2 in ppb (convert from µg/m³: 1 ppb SO2 = 2.62 µg/m³ at 25°C)
# df['SO2_ppb'] = df['SO2'] / 2.62
so2_breakpoints = [
    (0, 35, 0, 50),      # 0-35 ppb
    (36, 75, 51, 100),   # 36-75 ppb
    (76, 185, 101, 150), # 76-185 ppb
    (186, 304, 151, 200), # 186-304 ppb
    (305, 604, 201, 300), # 305-604 ppb
    (605, 804, 301, 400), # 605-804 ppb
    (805, 1004, 401, 500) # 805-1004 ppb
]

# CO in ppm (convert from mg/m³: 1 mg/m³ CO = 0.873 ppm at 25°C)
df['CO_ppm'] = df['CO'] * 0.873
co_breakpoints = [  # in ppm
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500)
]

# O3 in ppb (convert from µg/m³: 1 ppb O3 = 2.00 µg/m³ at 25°C)
# df['O3_ppb'] = df['O3'] / 2.00
o3_breakpoints = [
    (0, 54, 0, 50),      # 0-54 ppb
    (55, 70, 51, 100),   # 55-70 ppb
    (71, 85, 101, 150),  # 71-85 ppb
    (86, 105, 151, 200), # 86-105 ppb
    (106, 200, 201, 300) # 106-200 ppb
]

print("Calculating AQI for each pollutant...")

# Calculate AQI for each pollutant (using correct conversions)
df['AQI_PM25'] = df['PM2.5'].apply(lambda x: calculate_aqi_component(x, pm25_breakpoints))
df['AQI_PM10'] = df['PM10'].apply(lambda x: calculate_aqi_component(x, pm10_breakpoints))
df['AQI_NO2'] = (df['NO2'] / 1.88).apply(lambda x: calculate_aqi_component(x, no2_breakpoints))
df['AQI_SO2'] = (df['SO2'] / 2.62).apply(lambda x: calculate_aqi_component(x, so2_breakpoints))
df['AQI_CO'] = df['CO_ppm'].apply(lambda x: calculate_aqi_component(x, co_breakpoints))
df['AQI_O3'] = (df['O3'] / 2.00).apply(lambda x: calculate_aqi_component(x, o3_breakpoints))

# Overall AQI (maximum of all pollutants) - CORRECT
aqi_columns = ['AQI_PM25', 'AQI_PM10', 'AQI_NO2', 'AQI_SO2', 'AQI_CO', 'AQI_O3']
df['AQI'] = df[aqi_columns].max(axis=1)

# Identify which pollutant drives the AQI
def get_aqi_driver(row):
    aqi_values = {col: row[col] for col in aqi_columns}
    max_pollutant = max(aqi_values, key=aqi_values.get)
    # Extract pollutant name from column name
    pollutant = max_pollutant.replace('AQI_', '')
    if pollutant == 'PM25':
        return 'PM2.5'
    return pollutant

df['AQI_Driver'] = df.apply(get_aqi_driver, axis=1)

print(f"Top pollutants driving AQI:")
print(df['AQI_Driver'].value_counts().head())

# ============================================
# 5. AQI CATEGORIZATION
# ============================================
print("\n5. CATEGORIZING AQI LEVELS...")
print("-" * 40)

def categorize_aqi(aqi_value):
    """Categorize AQI into health risk levels"""
    if aqi_value <= 50:
        return 'Good'
    elif aqi_value <= 100:
        return 'Moderate'
    elif aqi_value <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi_value <= 200:
        return 'Unhealthy'
    elif aqi_value <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

df['AQI_Category'] = df['AQI'].apply(categorize_aqi)

# Map to health impacts
health_impact_map = {
    'Good': 'Little or no risk',
    'Moderate': 'Acceptable air quality',
    'Unhealthy for Sensitive Groups': 'Mild health effects for sensitive people',
    'Unhealthy': 'Health effects for everyone',
    'Very Unhealthy': 'Serious health effects',
    'Hazardous': 'Emergency conditions'
}

df['Health_Impact'] = df['AQI_Category'].map(health_impact_map)

print("AQI Categories Distribution:")
category_dist = df['AQI_Category'].value_counts()
print(category_dist)
print(f"\nAverage AQI: {df['AQI'].mean():.1f}")
print(f"Minimum AQI: {df['AQI'].min():.1f}")
print(f"Maximum AQI: {df['AQI'].max():.1f}")
print(f"\nMost frequent AQI driver: {df['AQI_Driver'].mode()[0]}")

# ============================================
# 6. OUTLIER HANDLING
# ============================================
print("\n6. HANDLING OUTLIERS...")
print("-" * 40)

# Select numerical columns
numerical_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 
                  'Temperature', 'Humidity', 'Wind Speed', 'AQI']

# IQR method for outlier detection and capping
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("Outliers capped using IQR method")

# ============================================
# 7. FEATURE SCALING
# ============================================
print("\n7. FEATURE SCALING...")
print("-" * 40)

# Features to scale
features_to_scale = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 
                     'Temperature', 'Humidity', 'Wind Speed']

# Standardization
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[features_to_scale]),
    columns=[f'{col}_scaled' for col in features_to_scale]
)

# Min-Max Normalization
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    minmax_scaler.fit_transform(df[features_to_scale]),
    columns=[f'{col}_normalized' for col in features_to_scale]
)

# Combine with original
df = pd.concat([df, df_scaled, df_normalized], axis=1)

print(f"Added {len(features_to_scale)} scaled features")

# ============================================
# 8. PREPARE DATA FOR MODELING
# ============================================
print("\n8. PREPARING DATA FOR MODELING...")
print("-" * 40)

# For regression: Predict AQI
X_reg = df[features_to_scale]
y_reg = df['AQI']

# For classification: Predict AQI Category
le = LabelEncoder()
y_clf = le.fit_transform(df['AQI_Category'])
X_clf = X_reg.copy()

# Train-test split (80/20) with stratification for classification
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42, stratify=df['AQI_Category']
)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print(f"Regression - Train: {X_reg_train.shape}, Test: {X_reg_test.shape}")
print(f"Classification - Train: {X_clf_train.shape}, Test: {X_clf_test.shape}")

# ============================================
# 9. SAVE PROCESSED DATA
# ============================================
print("\n9. SAVING PROCESSED DATA...")
print("-" * 40)

# Save preprocessed data
df.to_csv('air_quality_preprocessed.csv', index=False)

# Save splits for modeling
import joblib
modeling_data = {
    'X_reg_train': X_reg_train, 'X_reg_test': X_reg_test,
    'y_reg_train': y_reg_train, 'y_reg_test': y_reg_test,
    'X_clf_train': X_clf_train, 'X_clf_test': X_clf_test,
    'y_clf_train': y_clf_train, 'y_clf_test': y_clf_test,
    'features': features_to_scale,
    'label_encoder': le,
    'aqi_driver_stats': df['AQI_Driver'].value_counts().to_dict()
}

joblib.dump(modeling_data, 'modeling_data.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("✓ Saved preprocessed data to 'air_quality_preprocessed.csv'")
print("✓ Saved modeling data to 'modeling_data.joblib'")

# ============================================
# 10. INITIAL ANALYSIS
# ============================================
print("\n10. INITIAL ANALYSIS...")
print("-" * 40)

# Create pollutant summary
pollutant_summary = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].describe()
print("\nPollutant Concentration Summary (µg/m³ except CO in mg/m³):")
print(pollutant_summary.round(2))

# AQI driver analysis
print("\nAQI Driver Analysis:")
driver_stats = df['AQI_Driver'].value_counts()
for pollutant, count in driver_stats.items():
    percentage = (count / len(df)) * 100
    print(f"{pollutant}: {count} records ({percentage:.1f}%)")

# Save analysis
pollutant_summary.to_csv('pollutant_summary.csv')
print("✓ Saved pollutant summary to 'pollutant_summary.csv'")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("PREPROCESSING COMPLETE - SUMMARY")
print("="*60)

print(f"\nOriginal data shape: {df_original.shape}")
print(f"Processed data shape: {df.shape}")
print(f"Missing values handled: ✓")
print(f"AQI calculated correctly: ✓")
print(f"Top pollutant identified: {df['AQI_Driver'].mode()[0]}")
print(f"AQI categories created: ✓")
print(f"Outliers handled: ✓")
print(f"Features scaled: ✓")
print(f"Train-test splits created: ✓")
print(f"\nFiles created:")
print("  - air_quality_preprocessed.csv")
print("  - modeling_data.joblib")
print("  - data_limitations_report.txt")
print("  - pollutant_summary.csv")
print("\n✅ NEXT: Run EDA_Analysis.py")