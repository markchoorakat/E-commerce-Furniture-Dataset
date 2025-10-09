import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import hstack
import re

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
try:
    df = pd.read_csv('ecommerce_furniture_dataset_2024.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'ecommerce_furniture_dataset_2024.csv' was not found.")
    exit()

# -----------------------------------------------------------------------------
# 2. Data Preprocessing
print("\n--- Starting Data Preprocessing ---")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


def clean_price(price_str):
    if isinstance(price_str, str):
        cleaned_str = re.sub(r'[^\d.]', '', price_str)
        if cleaned_str: return float(cleaned_str)
    elif isinstance(price_str, (int, float)):
        return float(price_str)
    return np.nan


df['price'] = df['price'].apply(clean_price)
df['originalprice'] = df['originalprice'].apply(clean_price)

df['originalprice'] = df['originalprice'].fillna(df['price'])
df['tagtext'] = df['tagtext'].fillna('No Tag')
df.dropna(subset=['price', 'producttitle'], inplace=True)
df['sold'] = pd.to_numeric(df['sold'], errors='coerce').fillna(0).astype(int)
print("Preprocessing complete.")
# -----------------------------------------------------------------------------

# 3. Exploratory Data Analysis (EDA) with Plots
print("\n--- Starting Exploratory Data Analysis ---")

print("Showing plot for the distribution of items sold...")
plt.figure(figsize=(12, 6))
sns.histplot(df['sold'], bins=50, kde=True)
plt.title('Distribution of Number of Items Sold (Raw)')
plt.xlabel('Number of Items Sold')
plt.ylabel('Frequency')
plt.xlim(0, df['sold'].quantile(0.95))
plt.show()

print("Showing scatter plot for Price vs. Items Sold...")
plt.figure(figsize=(12, 6))
sns.scatterplot(x='price', y='sold', data=df, alpha=0.5)
plt.title('Price vs. Items Sold')
plt.xlabel('Price ($)')
plt.ylabel('Number of Items Sold')
plt.ylim(0, df['sold'].quantile(0.98))
plt.show()

print("EDA complete.")
# -----------------------------------------------------------------------------

# 4. Feature Engineering
print("\n--- Starting Feature Engineering ---")

df['discount_percentage'] = np.where(
    df['originalprice'] > 0,
    ((df['originalprice'] - df['price']) / df['originalprice']) * 100,
    0
).clip(min=0)

print("Showing plots comparing original and log-transformed target variable distributions...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(df['sold'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Original Distribution of "sold" (Capped at 95th Percentile)')
axes[0].set_xlabel('Number of Items Sold')
axes[0].set_xlim(0, df['sold'].quantile(0.95))
y_log_transformed = np.log1p(df['sold'])
sns.histplot(y_log_transformed, bins=50, kde=True, ax=axes[1])
axes[1].set_title('Log-Transformed Distribution of "sold"')
axes[1].set_xlabel('log(sold + 1)')
plt.tight_layout()
plt.show()

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_title_tfidf = tfidf_vectorizer.fit_transform(df['producttitle'])
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
X_tag_onehot = onehot_encoder.fit_transform(df[['tagtext']])
X_numerical = df[['price', 'discount_percentage']].values
X_combined = hstack([X_title_tfidf, X_tag_onehot, X_numerical])
y = y_log_transformed
print("Feature Engineering complete.")
# -----------------------------------------------------------------------------

# 5. Model Training & Tuning
print("\n--- Starting Model Training & Tuning ---")
X_train, X_test, y_train_log, y_test_log = train_test_split(X_combined, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_log)
param_grid = {
    'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train_log)
best_rf_model = grid_search.best_estimator_
print(f"Best Hyperparameters found: {grid_search.best_params_}")
print("Model Training complete.")
# -----------------------------------------------------------------------------

# 6. Model Evaluation with Plots
print("\n--- Evaluating Models ---")
y_pred_log_lr = lr_model.predict(X_test)
y_pred_log_rf_tuned = best_rf_model.predict(X_test)
y_test_orig = np.expm1(y_test_log)
y_pred_orig_lr = np.expm1(y_pred_log_lr)
y_pred_orig_rf_tuned = np.expm1(y_pred_log_rf_tuned)

print("Showing plot of Actual vs. Predicted values for both models...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.scatterplot(x=y_test_orig, y=y_pred_orig_lr, alpha=0.6, ax=axes[0])
axes[0].set_title('Linear Regression: Actual vs. Predicted')
axes[0].set_xlabel('Actual Sold')
axes[0].set_ylabel('Predicted Sold')
axes[0].plot([0, max(y_test_orig)], [0, max(y_test_orig)], color='red', linestyle='--')
sns.scatterplot(x=y_test_orig, y=y_pred_orig_rf_tuned, alpha=0.6, ax=axes[1])
axes[1].set_title('Tuned Random Forest: Actual vs. Predicted')
axes[1].set_xlabel('Actual Sold')
axes[1].set_ylabel('Predicted Sold')
axes[1].plot([0, max(y_test_orig)], [0, max(y_test_orig)], color='red', linestyle='--')
plt.tight_layout()
plt.show()

print("Showing plot of Top 20 Feature Importances from the Random Forest model...")
try:
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    ohe_features = onehot_encoder.get_feature_names_out(['tagtext'])
    numerical_features = ['price', 'discount_percentage']
    all_features = list(tfidf_features) + list(ohe_features) + list(numerical_features)

    importances = best_rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': all_features, 'importance': importances})
    top_20_features = feature_importance_df.nlargest(20, 'importance')

    plt.figure(figsize=(12, 8))
    # --- FIX: Updated syntax to resolve FutureWarning ---
    sns.barplot(x='importance', y='feature', data=top_20_features, hue='feature', palette='viridis', legend=False)
    # --- End of Fix ---
    plt.title('Top 20 Most Important Features (Tuned Random Forest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not generate feature importance plot. Error: {e}")

mse_lr = mean_squared_error(y_test_orig, y_pred_orig_lr)
r2_lr = r2_score(y_test_orig, y_pred_orig_lr)
mse_rf_tuned = mean_squared_error(y_test_orig, y_pred_orig_rf_tuned)
r2_rf_tuned = r2_score(y_test_orig, y_pred_orig_rf_tuned)

print("\n--- Model Performance Results (Evaluated on Original Scale) ---")
print("\n1. Linear Regression Model:")
print(f"   - Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"   - R-squared (R2) Score: {r2_lr:.3f}")
print("\n2. Tuned Random Forest Model:")
print(f"   - Mean Squared Error (MSE): {mse_rf_tuned:.2f}")
print(f"   - R-squared (R2) Score: {r2_rf_tuned:.3f}")
# -----------------------------------------------------------------------------

# 7. Conclusion
print("\n--- Conclusion ---")
if r2_rf_tuned > r2_lr:
    print("After tuning, the Random Forest model now outperforms the Linear Regression model.")
else:
    print("Even after tuning, the Linear Regression model remains the better choice for this problem.")
print("\nScript execution finished.")
# -----------------------------------------------------------------------------

