import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. DATA SIMULATION & INITIAL SETUP ---
print("--- 1. DATA SIMULATION & INITIAL SETUP ---")

# Simulate sensor logs for 5 equipment units over 300 days (hourly logs)
np.random.seed(42)
days = 300
hours = days * 24
equipment_ids = [101, 102, 103, 104, 105]
num_records = len(equipment_ids) * hours

# Generate base data structure
data = {
    'equipment_id': np.repeat(equipment_ids, hours),
    'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=hours, freq='H').tolist() * len(equipment_ids)),
    # Generate sensor data (add some noise for realism)
    'temperature': np.random.normal(loc=70, scale=5, size=num_records),
    'vibration': np.random.normal(loc=15, scale=3, size=num_records),
    'speed': np.random.normal(loc=1500, scale=100, size=num_records),
    'torque': np.random.normal(loc=40, scale=5, size=num_records),
    'is_failed': np.zeros(num_records) # 1 if failure occurs at this timestamp, 0 otherwise
}

df = pd.DataFrame(data)

# Inject failures (LOGICAL ERROR CORRECTED): Mark the exact timestamp of failure
# Equipment 102 fails on a specific timestamp (approx. day 150)
df.loc[(df['equipment_id'] == 102) & (df['timestamp'] == '2024-06-01 00:00:00'), 'is_failed'] = 1
# Equipment 105 fails on another specific timestamp (approx. day 250)
df.loc[(df['equipment_id'] == 105) & (df['timestamp'] == '2024-09-06 00:00:00'), 'is_failed'] = 1

print(f"Total records: {len(df)}")
print(f"Total failures (terminal events): {df['is_failed'].sum()}\n")


# --- 2. FEATURE ENGINEERING (Pandas & NumPy) ---
print("--- 2. FEATURE ENGINEERING (Pandas & NumPy) ---")

# Sort by equipment and time for correct rolling calculations
df = df.sort_values(by=['equipment_id', 'timestamp']).reset_index(drop=True)

# A. Rolling Window Features (72-hour average and standard deviation)
WINDOW = 72 # 72 hours = 3 days

for col in ['temperature', 'vibration', 'speed', 'torque']:
    # Calculate Rolling Mean (Average)
    df[f'{col}_avg_72h'] = df.groupby('equipment_id')[col].transform(
        lambda x: x.rolling(window=WINDOW, min_periods=1).mean()
    )
    # Calculate Rolling Standard Deviation (Volatility)
    df[f'{col}_std_72h'] = df.groupby('equipment_id')[col].transform(
        lambda x: x.rolling(window=WINDOW, min_periods=1).std()
    )

# B. Create Target Variable: Predict failure in the next 7 days (168 hours)
FAIL_WINDOW = 168 # 7 days * 24 hours

# LOGIC: Apply a rolling max (over the future 168 hours) to the 'is_failed' column
# This finds if a failure (1) occurs at any point in the next 168 timestamps.
# closed='left' ensures the current timestamp is excluded from the window.
# .shift(-FAIL_WINDOW) moves the future window result back to the current row.

# Step 1: Calculate if a failure occurs in the window AFTER the current row
df['future_failure_flag'] = df.groupby('equipment_id')['is_failed'].transform(
    lambda x: x.rolling(window=FAIL_WINDOW, closed='left').max().shift(-FAIL_WINDOW)
)

# Target Variable: 1 if failure is predicted in next 7 days, 0 otherwise
df['target_failure_7d'] = df['future_failure_flag'].fillna(0).astype(int)


# C. Final Cleanup
# Drop initial rows where rolling features are incomplete (first WINDOW-1 rows)
df = df.dropna().reset_index(drop=True)

print(f"Records after feature engineering and cleaning: {len(df)}")
print(f"Target distribution (1=Failure): {df['target_failure_7d'].value_counts(normalize=True).round(4)}\n")


# --- 3. EXPLORATORY DATA ANALYSIS (Matplotlib & Seaborn) ---
print("--- 3. EXPLORATORY DATA ANALYSIS (Matplotlib & Seaborn) ---")
sns.set_style("whitegrid")

# A. Feature Distribution (Box Plot)
plt.figure(figsize=(12, 5))
# Use the engineered feature and the target variable
sns.boxplot(x='target_failure_7d', y='temperature_avg_72h', data=df, palette='viridis')
plt.title('72-Hour Avg Temperature: Normal vs. Near Failure')
plt.xlabel('Failure in Next 7 Days (0=No, 1=Yes)')
plt.ylabel('72-Hour Avg Temperature')
plt.show() 

# B. Correlation Heatmap
# Includes both raw features and engineered rolling features
feature_cols = [col for col in df.columns if col.endswith('_72h') or col in ['speed', 'torque', 'vibration', 'temperature']]
correlation_matrix = df[feature_cols + ['target_failure_7d']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Feature Correlation Heatmap')
plt.show() 


# --- 4. MODELING & EVALUATION (Scikit-learn) ---
print("\n--- 4. MODELING & EVALUATION ---")

# Define Features (X) and Target (y)
# ERROR CORRECTED: Explicitly define all engineered and raw sensor features for X
engineered_features = [col for col in df.columns if col.endswith('_72h')]
raw_features = ['speed', 'torque', 'vibration', 'temperature']
final_features = list(set(engineered_features + raw_features)) # Combine and ensure uniqueness

X = df[final_features]
y = df['target_failure_7d']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Stratify=y ensures both train and test sets have the same proportion of the rare "1" class.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# Train a Random Forest Classifier (using class_weight='balanced' to handle imbalance)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation
print("\nClassification Report (Key Metrics: Precision, Recall, F1-Score for '1'):")
print(classification_report(y_test, predictions))

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Failure (0)', 'Failure (1)'],
            yticklabels=['No Failure (0)', 'Failure (1)'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

print("\nProject pipeline complete. The Confusion Matrix is the most important output for this imbalanced problem.")