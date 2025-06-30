# House Price Prediction Using Linear Regression and Gradient Boosting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# 1. Load dataset
df = pd.read_csv('kc_house_data.csv')

# 2. Select relevant columns
df = df[['price', 'sqft_living', 'bedrooms', 'zipcode']]  # treating 'zipcode' as location

# 3. Drop rows with missing values
df = df.dropna()

# 4. Define features and target
X = df[['sqft_living', 'bedrooms', 'zipcode']]
y = df['price']

# 5. Define column types
numeric_features = ['sqft_living', 'bedrooms']
categorical_features = ['zipcode']

# 6. Create preprocessing steps
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Linear Regression model
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

# 9. Gradient Boosting model
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])
gb_pipeline.fit(X_train, y_train)
gb_preds = gb_pipeline.predict(X_test)

# 10. Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} Results:")
    print(f"  MAE:  {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}\n")

evaluate_model(y_test, lr_preds, "Linear Regression")
evaluate_model(y_test, gb_preds, "Gradient Boosting")

# 11. Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=lr_preds, alpha=0.4)
plt.title('Linear Regression Predictions')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=gb_preds, alpha=0.4, color='orange')
plt.title('Gradient Boosting Predictions')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

plt.tight_layout()
plt.show()
