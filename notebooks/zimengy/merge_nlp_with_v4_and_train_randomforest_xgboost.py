import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. 加载数据
path_main = r'C:\Users\lola\github\Airbnb-Renting-Optimizer\data_processed\Chicago_Airbnb_Master_v4.csv'
path_nlp = r'C:\Users\lola\github\Airbnb-Renting-Optimizer\data_processed\listings_with_nlp_features.csv'

df_main = pd.read_csv(path_main)
df_nlp = pd.read_csv(path_nlp)

# 2. 合并数据 (以 listing_id 为基准)
# 确保 ID 列名一致，提取 NLP 中的 Topic 列
nlp_cols = ['id', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'topic_7']
df = pd.merge(df_main, df_nlp[nlp_cols], left_on='listing_id', right_on='id', how='left')

# 3. 特征筛选
# 目标变量：log_revenue
target = 'log_revenue'

# 排除非特征列（ID、文本、以及会导致数据泄露的原始收入/价格列）
cols_to_drop = [
    'listing_id', 'id', 'description', 'tokens_final', 'price', 
    'estimated_annual_revenue', 'log_revenue', 'neighbourhood_cleansed' # 经纬度已提供地理信息
]

# 4. 预处理：处理分类变量和缺失值
# 将 'f'/'t' 或 True/False 转为 0/1
bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'has_availability']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0)

# 处理 room_type 等剩余类别特征 (One-hot encoding)
X = df.drop(columns=cols_to_drop)
X = pd.get_dummies(X, drop_first=True)

# 处理缺失值 (用中位数填充)
X = X.fillna(X.median())
y = df[target].fillna(df[target].median())

# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 模型 A: Random Forest ---
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# --- 模型 B: XGBoost ---
print("Training XGBoost...")
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# 6. 结果对比
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

# 7. 特征重要性可视化 (以 XGBoost 为例)
plt.figure(figsize=(10, 8))
feat_importances = pd.Series(xgb.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh', color='skyblue')
plt.title('Top 20 Important Features for Revenue Prediction')
plt.xlabel('Importance Score')
plt.show()
