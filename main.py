import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

file_path = 'Fuel_cell_performance_data-Full.csv'
dataset = pd.read_csv(file_path)

dataset.head(), dataset.columns

X = dataset.drop(['A', 'B', 'C', 'D', 'E'], axis=1)
y = dataset['Target5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regressor (SVR)": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor(),
    "Lasso Regression": Lasso()
}
results = {}
predictions = {}
for model_name, model in models.items():
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred
    
    r2 = r2_score(y_test, y_pred)
    results[model_name] = r2
best_model_name = max(results, key=results.get)
best_r2_score = results[best_model_name]


# Results and the best model
print("Model Performance:")
for model, score in results.items():
    print(f"{model}: R² = {score:.4f}")
print(f"\nBest Model: {best_model_name} with R² = {best_r2_score:.4f}")


# Scatter plots for Actual vs. Predicted values for each model
plt.figure(figsize=(12, 8))
for i, (model_name, y_pred) in enumerate(predictions.items()):
    plt.subplot(2, 4, i+1)
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
    plt.title(f"{model_name}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.tight_layout()
plt.suptitle("Actual vs. Predicted Values for All Models", y=1.02, fontsize=16)
plt.show()

# Residual plots for the best model
best_model = models[best_model_name]
best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)
residuals = y_test - best_pred

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='yellow', bins=30)
plt.title(f"Residuals Distribution for {best_model_name}")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.axvline(0, color='blue', linestyle='--', label='Zero Residual')
plt.legend()
plt.show()
