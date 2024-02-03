import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
data_set = pd.read_csv('Data/classification_dataset.csv')

# Check basic statistics and missing values
basic_stats = data_set.describe()
missing_values = data_set.isnull().sum()
print(basic_stats, missing_values)

# StandardScaler object
scaler = StandardScaler()

# Standardize the features excluding the label column
features_scaled = scaler.fit_transform(data_set.iloc[:, :-1])

# Create a DataFrame with standardized features
data_set_scaled = pd.DataFrame(features_scaled, columns=data_set.columns[:-1])

# Add the labels back to the DataFrame
data_set_scaled['Label'] = data_set['Label']

def determine_train_test_ratio(last_name):
    import random
    value = len(last_name)
    random.seed(value)
    train_test_ratio = round(random.uniform(0.1, 0.8), 2)
    return train_test_ratio

last_name = "yılmaz"
train_test_ratio = determine_train_test_ratio(last_name)
print(f"Dear {last_name}, your train-test ratio is: {train_test_ratio}")

# Split the data into features (X) and labels (y)
X = data_set_scaled.iloc[:, :-1]  # Features
y = data_set_scaled['Label']  # Labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_test_ratio), random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear')  # SVM model with linear kernel
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest model with 100 trees
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate the accuracy score
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(accuracy_rf)

# Create and train the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gb_model.predict(X_test)

# Calculate the accuracy score
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(accuracy_gb)

# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_model.predict(X_test)

# Calculate the accuracy score
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(accuracy_dt)

# Create an array with model names and corresponding accuracy values
accuracy_dict = {
    'Model': ['SVM', 'Random Forest', 'Gradient Boosting', 'Decision Tree'],
    'Accuracy': [accuracy, accuracy_rf, accuracy_gb, accuracy_dt]
}

# Convert the dictionary to DataFrame and sort by accuracy values in descending order
accuracy_df = pd.DataFrame(accuracy_dict)
accuracy_df_sorted = accuracy_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print(accuracy_df_sorted)

# Save the Random Forest model in joblib format
model_path = 'model/random_forest_model.joblib'
joblib.dump(rf_model, model_path)
print(model_path)