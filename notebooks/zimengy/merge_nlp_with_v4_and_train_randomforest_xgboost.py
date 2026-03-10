import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load datasets
path_main = r'C:\Users\lola\github\Airbnb-Renting-Optimizer\data_processed\Chicago_Airbnb_Master_v4.csv'
path_nlp = r'C:\Users\lola\github\Airbnb-Renting-Optimizer\data_processed\listings_with_nlp_features.csv'

df_main = pd.read_csv(path_main)
df_nlp = pd.read_csv(path_nlp)

# 2. Merge datasets using listing_id
# Ensure consistent ID column names and extract topic features from the NLP dataset
nlp_cols = ['id', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'topic_7']
df = pd.merge(df_main, df_nlp[nlp_cols], left_on='listing_id', right_on='id', how='left')

# 3. Feature selection
# Target variable: log_revenue
target = 'log_revenue'

# Drop non-feature columns (IDs, text fields, and variables that could cause data leakage such as revenue/price)
cols_to_drop = [
    'listing_id', 'id', 'description', 'tokens_final', 'price', 
    'estimated_annual_revenue', 'log_revenue', 'neighbourhood_cleansed'  # Geographic info already captured by other location features
]

# 4. Preprocessing: handle categorical variables and missing values
# Convert 't'/'f' or True/False values to binary (0/1)
bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'has_availability']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0)

# Encode remaining categorical features such as room_type using one-hot encoding
X = df.drop(columns=cols_to_drop)
X = pd.get_dummies(X, drop_first=True)

# Handle missing values by filling with the median
X = X.fillna(X.median())
y = df[target].fillna(df[target].median())

# 5. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model A: Random Forest ---
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# --- Model B: XGBoost ---
print("Training XGBoost...")
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# 6. Model evaluation
def evaluate(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n[{model_name}] Results:")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

evaluate(y_test, rf_pred, "Random Forest")
evaluate(y_test, xgb_pred, "XGBoost")

# 7. Feature importance visualization (using XGBoost as an example)
plt.figure(figsize=(10, 8))
feat_importances = pd.Series(xgb.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh', color='skyblue')
plt.title('Top 20 Important Features for Revenue Prediction')
plt.xlabel('Importance Score')
plt.show()
