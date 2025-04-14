
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

df=pd.read_csv("insurance.csv")


print(df.info())
print(df.describe())
print(df.describe(include='object')) # included categorical features in the statistical description

# boxplot to visualize outliers
# Select numerical columns
numerical_cols = ['age', 'bmi', 'children', 'expenses']

# Create boxplots
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Outliers in {col}')
plt.tight_layout()
plt.show()



sns.set(style="darkgrid")
sns.pairplot(df)


# 1. Distribution of Target Variable (Expenses)
plt.figure(figsize=(8, 5))
sns.histplot(df['expenses'], kde=True, bins=30)
plt.title('Distribution of Insurance Expenses')
plt.xlabel('Expenses')
plt.ylabel('Frequency')
plt.show()

# 2. Correlation Heatmap (Numeric Features)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Between Numeric Features')
plt.show()

## Categorical Features vs Expenses (Boxplots)

# 1. Smoker vs Expenses
plt.figure(figsize=(6, 4))
sns.boxplot(x='smoker', y='expenses', data=df)
plt.title('Expenses by Smoking Status')
plt.show()

# 2. Sex vs Expenses
plt.figure(figsize=(6, 4))
sns.boxplot(x='sex', y='expenses', data=df)
plt.title('Expenses by Sex')
plt.show()

# 3. Region vs Expenses
plt.figure(figsize=(8, 5))
sns.boxplot(x='region', y='expenses', data=df)
plt.title('Expenses by Region')
plt.show()


# 4. Boxplot: Expenses by Number of Children
plt.figure(figsize=(6, 4))
sns.boxplot(x='children', y='expenses', data=df)
plt.title('Expenses by Number of Children')
plt.xlabel('Number of Children')
plt.ylabel('Expenses')
plt.show()

## Continuous Features vs Expenses (Scatterplots)

# 1. Age vs Expenses
plt.figure(figsize=(6, 4))
sns.scatterplot(x='age', y='expenses', data=df)
plt.title('Age vs Expenses')
plt.show()

# 2. BMI vs Expenses
plt.figure(figsize=(6, 4))
sns.scatterplot(x='bmi', y='expenses', data=df)
plt.title('BMI vs Expenses')
plt.show()

#: Highlight Smokers in BMI vs Expenses
plt.figure(figsize=(6, 4))
sns.scatterplot(x='bmi', y='expenses', hue='smoker', data=df)
plt.title('BMI vs Expenses Colored by Smoker Status')
plt.show()



# 1. Encode Categorical Features
df_encoded = pd.get_dummies(df, drop_first=True).astype(int)

# create interaction terms
sns.scatterplot(x=df_encoded['age'] * df_encoded['smoker_yes'], y=df_encoded['expenses'])
plt.title('Expenses vs Age * Smoker')
plt.xlabel('Age * Smoker')
plt.ylabel('Expenses')
plt.show()

sns.scatterplot(x=df_encoded['age'] * df_encoded['bmi'], y=df_encoded['expenses'])
plt.title('Expenses vs Age * BMI')
plt.xlabel('Age * BMI')
plt.ylabel('Expenses')
plt.show()

sns.scatterplot(x=df_encoded['age'] * df_encoded['children'], y=df_encoded['expenses'])
plt.title('Expenses vs Age * Children')
plt.xlabel('Age * Children')
plt.ylabel('Expenses')
plt.show()

sns.scatterplot(x=df_encoded['sex_male'] * df_encoded['smoker_yes'], y=df_encoded['expenses'])
plt.title('Expenses vs Sex * Smoker')
plt.xlabel('Sex * Smoker')
plt.ylabel('Expenses')
plt.show()

sns.boxplot(x=df_encoded['children'] * df_encoded['smoker_yes'], y=df_encoded['expenses'])
plt.title('Expenses by Children-Smoker Interaction')
plt.xlabel('Children * Smoker')
plt.ylabel('Expenses')
plt.show()



df['log_expenses'] = np.log1p(df['expenses'])

# age * smoker VS log_expenses
sns.scatterplot(x=df_encoded['age'] * df_encoded['smoker_yes'], y=df['log_expenses'])
plt.title('Log Expenses vs Age * Smoker')
plt.xlabel('Age * Smoker')
plt.ylabel('Log Expenses')
plt.show()

# age * bmi VS log_expenses
sns.scatterplot(x=df_encoded['age'] * df_encoded['bmi'], y=df['log_expenses'])
plt.title('Log Expenses vs Age * BMI')
plt.xlabel('Age * BMI')
plt.ylabel('Log Expenses')
plt.show()

# age * children VS log_expenses
sns.scatterplot(x=df_encoded['age'] * df_encoded['children'], y=df['log_expenses'])
plt.title('Log Expenses vs Age * Children')
plt.xlabel('Age * Children')
plt.ylabel('Log Expenses')
plt.show()

# sex_male * smoker_yes VS log_expenses
sns.scatterplot(x=df_encoded['sex_male'] * df_encoded['smoker_yes'], y=df['log_expenses'])
plt.title('Log Expenses vs Sex * Smoker')
plt.xlabel('Sex * Smoker')
plt.ylabel('Log Expenses')
plt.show()

# children * smoker_yes VS log expenses
sns.boxplot(x=df_encoded['children'] * df_encoded['smoker_yes'], y=df['log_expenses'])
plt.title('Log Expenses by Children-Smoker Interaction')
plt.xlabel('Children * Smoker')
plt.ylabel('Log Expenses')
plt.show()



## Preprocessing begins


## PREPROCESSING PIPELINE

# 1. Encode Categorical Features
df_encoded = pd.get_dummies(df, drop_first=True).astype(int)

# 2. Created Interaction Features (from EDA insights)
df_encoded['bmi_smoker'] = df_encoded['bmi'] * df_encoded['smoker_yes']
df_encoded['age_smoker'] = df_encoded['age'] * df_encoded['smoker_yes']
df_encoded['age_bmi'] = df_encoded['age'] * df_encoded['bmi']
df_encoded['sex_smoker'] = df_encoded['sex_male'] * df_encoded['smoker_yes']
df_encoded['children_smoker'] = df_encoded['children'] * df_encoded['smoker_yes']

# 3. Log-transform Target (used for models like KNN and Linear Regression only)
df_encoded['log_expenses'] = np.log1p(df_encoded['expenses'])  # use np.expm1() to reverse

# 4. Define Features and Targets
X = df_encoded.drop(columns=['expenses', 'log_expenses'])  # independent variables
y_raw = df_encoded['expenses']          # original target
y_log = df_encoded['log_expenses']      # log-transformed target

# 5. Train-Test Split
from sklearn.model_selection import train_test_split

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.2, random_state=42
)
_, _, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# 6. Scale Features (for scaling-sensitive models like KNN, LR)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 7. Check shapes and previews
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("Encoded DataFrame shape:", df_encoded.shape)
print("X sample:\n", X.head().T)
print("y_raw sample:\n", y_raw.head())
print("y_log sample:\n", y_log.head())



# Model dictionary with their settings
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_log,
        'y_test': y_test_raw,  # compare with original after exp transform
        'log_target': True
    },
    'KNN': {
        'model': KNeighborsRegressor(n_neighbors=5),
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_log,
        'y_test': y_test_raw,
        'log_target': True
    },
#     'Decision Tree': {
#         'model': DecisionTreeRegressor(random_state=42),
#         'X_train': X_train_raw,
#         'X_test': X_test_raw,
#         'y_train': y_train_raw,
#         'y_test': y_test_raw,
#         'log_target': False
#     },
#     'Random Forest': {
#         'model': RandomForestRegressor(random_state=42),
#         'X_train': X_train_raw,
#         'X_test': X_test_raw,
#         'y_train': y_train_raw,
#         'y_test': y_test_raw,
#         'log_target': False
#     },
#     'Gradient Boosting': {
#         'model': GradientBoostingRegressor(random_state=42),
#         'X_train': X_train_raw,
#         'X_test': X_test_raw,
#         'y_train': y_train_raw,
#         'y_test': y_test_raw,
#         'log_target': False
#     },
#     'XGBoost': {
#         'model': XGBRegressor(random_state=42, verbosity=0),
#         'X_train': X_train_raw,
#         'X_test': X_test_raw,
#         'y_train': y_train_raw,
#         'y_test': y_test_raw,
#         'log_target': False
#     }
 }

# Evaluation function
def evaluate_model(name, model, X_train, X_test, y_train, y_test, log_target=False):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Reverse log transform if needed
    if log_target:
        y_pred = np.expm1(y_pred)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "Model": name,
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    }

# Run training and evaluation for all models
results = []

for name, config in models.items():
    result = evaluate_model(
        name=name,
        model=config['model'],
        X_train=config['X_train'],
        X_test=config['X_test'],
        y_train=config['y_train'],
        y_test=config['y_test'],
        log_target=config['log_target']
    )
    results.append(result)

# Display results
results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
print(results_df)



# Step 1: Remove interaction terms
interaction_cols = ['bmi_smoker', 'age_smoker', 'age_bmi', 'sex_smoker', 'children_smoker']
X_base = df_encoded.drop(columns=['expenses', 'log_expenses'] + interaction_cols)

# Step 2: Define log-transformed target
y_log = df_encoded['log_expenses']

# Step 3: Train-test split for tree models (no interaction terms)

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_base, y_log, test_size=0.2, random_state=42
)

tree_models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0)
}

# Step 5: Evaluation function (back-transform target for fair comparison)
def evaluate_tree_model(name, model, X_train, X_test, y_train_log, y_test_log):
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)

    # Reverse log transformation to get predictions on original scale
    y_pred_raw = np.expm1(y_pred_log)
    y_test_raw = np.expm1(y_test_log)

    r2 = r2_score(y_test_raw, y_pred_raw)
    mae = mean_absolute_error(y_test_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))

    return {
        "Model": name,
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    }

# Step 6: Run training & evaluation
tree_results = []

for name, model in tree_models.items():
    result = evaluate_tree_model(
        name=name,
        model=model,
        X_train=X_train_tree,
        X_test=X_test_tree,
        y_train_log=y_train_tree,
        y_test_log=y_test_tree
    )
    tree_results.append(result)

# Step 7: Display results
tree_results_df = pd.DataFrame(tree_results).sort_values(by='R2 Score', ascending=False)
print(tree_results_df)

"""Here, I trained tree models on raw dataset without interaction terms but also on raw target variable (skewed target)"""

# Step 1: Define base X (no interaction terms)
interaction_cols = ['bmi_smoker', 'age_smoker', 'age_bmi', 'sex_smoker', 'children_smoker']
X_base = df_encoded.drop(columns=['expenses', 'log_expenses'] + interaction_cols)

# Step 2: Define raw target
y_raw = df_encoded['expenses']

# Step 3: Train-test split

X_train_tree_raw, X_test_tree_raw, y_train_tree_raw, y_test_tree_raw = train_test_split(
    X_base, y_raw, test_size=0.2, random_state=42
)

# Step 4: Define tree models

tree_models_raw_y = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0)
}

# Step 5: Evaluation function (no transformation needed)
def evaluate_tree_model_raw(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "Model": name,
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    }

# Step 6: Train and evaluate
tree_results_raw_y = []

for name, model in tree_models_raw_y.items():
    result = evaluate_tree_model_raw(
        name=name,
        model=model,
        X_train=X_train_tree_raw,
        X_test=X_test_tree_raw,
        y_train=y_train_tree_raw,
        y_test=y_test_tree_raw
    )
    tree_results_raw_y.append(result)

# Step 7: Display results
tree_results_raw_y_df = pd.DataFrame(tree_results_raw_y).sort_values(by='R2 Score', ascending=False)
print(tree_results_raw_y_df)


# Define refined search grid
grid_params = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 6, 10],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'subsample': [0.9, 1.0],
    'max_features': ['sqrt', 'log2']
}

# Set up GridSearchCV
gb_grid = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=grid_params,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Fit on training data (no log transform)
gb_grid.fit(X_train_tree, y_train_tree_raw)

# Output results
print(" Gradient Boosting Best Parameters:", gb_grid.best_params_)
print(" Gradient Boosting Best R² (CV):", gb_grid.best_score_)



# XGBoost: Define refined search grid
grid_params_xgb = {
    'n_estimators': [50, 150, 200],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 6, 10],
    'subsample': [0.9, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Set up GridSearchCV for XGBoost
xgb_grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42, verbosity=0),
    param_grid=grid_params_xgb,
    scoring='r2',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit XGBoost model on training data (raw predictors and raw target)
xgb_grid.fit(X_train_tree, y_train_tree_raw)

# Output best results for XGBoost
print("XGBoost Best Parameters:", xgb_grid.best_params_)
print("XGBoost Best R² Score:", xgb_grid.best_score_)

# Random Forest: Define refined search grid

grid_params_rf = {
    'n_estimators': [50, 150, 200],
    'max_depth': [3, 6, 10],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# Set up GridSearchCV for Random Forest
rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=grid_params_rf,
    scoring='r2',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit Random Forest model on training data
rf_grid.fit(X_train_tree, y_train_tree_raw)

# Output best results for Random Forest
print("Random Forest Best Parameters:", rf_grid.best_params_)
print("Random Forest Best R² Score:", rf_grid.best_score_)

best_gb_grid = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=4,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=150,
    subsample=1.0,
    random_state=42
)

best_gb_grid.fit(X_train_tree, y_train_tree_raw)

# eveluate
y_pred_grid = best_gb_grid.predict(X_test_tree)

r2 = r2_score(y_test_tree_raw, y_pred_grid)
mae = mean_absolute_error(y_test_tree_raw, y_pred_grid)
rmse = np.sqrt(mean_squared_error(y_test_tree_raw, y_pred_grid))

print("\nGradientBoosting Final Model Evaluation:")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")


best_rf = RandomForestRegressor(
    random_state=42,
    max_depth=6,
    max_features='log2',
    min_samples_leaf=1,
    min_samples_split=4,
    n_estimators=150
)

# Retrain on the full training set
best_rf.fit(X_train_tree, y_train_tree_raw)

# Predict on the test set
y_pred_rf = best_rf.predict(X_test_tree)

# Evaluate performance
r2_rf = r2_score(y_test_tree_raw, y_pred_rf)
mae_rf = mean_absolute_error(y_test_tree_raw, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_tree_raw, y_pred_rf))

print("\nRandom Forest Final Model Evaluation:")
print(f"Test R²: {r2_rf:.4f}")
print(f"Test MAE: {mae_rf:.2f}")
print(f"Test RMSE: {rmse_rf:.2f}")

best_xgb = XGBRegressor(
    random_state=42,
    colsample_bytree=1.0,
    learning_rate=0.07,
    max_depth=3,
    n_estimators=150,
    subsample=1.0,
    verbosity=0
)

# Retrain on the full training set
best_xgb.fit(X_train_tree, y_train_tree_raw)

# Predict on the test set
y_pred_xgb = best_xgb.predict(X_test_tree)

# Evaluate performance
r2_xgb = r2_score(y_test_tree_raw, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test_tree_raw, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test_tree_raw, y_pred_xgb))

print("\nXGBoost Final Model Evaluation:")
print(f"Test R²: {r2_xgb:.4f}")
print(f"Test MAE: {mae_xgb:.2f}")
print(f"Test RMSE: {rmse_xgb:.2f}")

"""Visualizing Feature importance across all three Top Models"""

import matplotlib.pyplot as plt
import pandas as pd


features = X_train_tree.columns

# Create DataFrames for feature importances for each model
df_xgb = pd.DataFrame({
    'Feature': features,
    'Importance': best_xgb.feature_importances_
}).sort_values(by='Importance', ascending=True)

df_gb = pd.DataFrame({
    'Feature': features,
    'Importance': best_gb_grid.feature_importances_
}).sort_values(by='Importance', ascending=True)

df_rf = pd.DataFrame({
    'Feature': features,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=True)

# Create a figure with three subplots for side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# XGBoost Feature Importance Plot
axes[0].barh(df_xgb['Feature'], df_xgb['Importance'], color='teal')
axes[0].set_title("XGBoost Feature Importances")
axes[0].set_xlabel("Importance")

# Gradient Boosting Feature Importance Plot
axes[1].barh(df_gb['Feature'], df_gb['Importance'], color='darkorange')
axes[1].set_title("Gradient Boosting Feature Importances")
axes[1].set_xlabel("Importance")

# Random Forest Feature Importance Plot
axes[2].barh(df_rf['Feature'], df_rf['Importance'], color='purple')
axes[2].set_title("Random Forest Feature Importances")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.show()

import joblib

joblib.dump(best_xgb, 'xgb_model.pkl')
print("Final Model saved as 'xgb_model.pkl'")

loaded_model = joblib.load('xgb_model.pkl')