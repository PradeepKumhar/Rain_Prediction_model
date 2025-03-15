from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import joblib
from data_processing import load_and_clean_data

# 🔹 Data Load karo
data, scaler = load_and_clean_data("dataset.csv")

# 🔹 Features aur Target alag karo
X = data.drop(columns=['Rain_Today'])
y = data['Rain_Today']

# 🔹 Feature Selection: Best 5 features choose karo
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# 🔹 Data Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 🔹 Class Imbalance Fix (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 🔹 Hyperparameter Tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  
    'solver': ['liblinear', 'lbfgs']
}
grid_search = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

# 🔹 Best Model Save karo
best_model = grid_search.best_estimator_
joblib.dump(best_model, "rain_prediction_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("✅ Model Training Complete with Feature Selection, Class Balancing & Hyperparameter Tuning")
