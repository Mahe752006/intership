import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate synthetic Bank Marketing-like dataset
np.random.seed(42)

# Simulate features similar to the Bank Marketing dataset
data_size = 1000
synthetic_data = pd.DataFrame({
    'age': np.random.randint(18, 95, size=data_size),
    'job': np.random.choice(['admin.', 'blue-collar', 'entrepreneur', 'management', 'retired', 'student'], size=data_size),
    'marital': np.random.choice(['married', 'single', 'divorced'], size=data_size),
    'education': np.random.choice(['primary', 'secondary', 'tertiary'], size=data_size),
    'default': np.random.choice(['yes', 'no'], size=data_size, p=[0.1, 0.9]),
    'balance': np.random.randint(-2000, 50000, size=data_size),
    'housing': np.random.choice(['yes', 'no'], size=data_size, p=[0.7, 0.3]),
    'loan': np.random.choice(['yes', 'no'], size=data_size, p=[0.2, 0.8]),
    'contact': np.random.choice(['cellular', 'telephone'], size=data_size),
    'day': np.random.randint(1, 31, size=data_size),
    'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], size=data_size),
    'duration': np.random.randint(1, 3000, size=data_size),
    'campaign': np.random.randint(1, 50, size=data_size),
    'pdays': np.random.choice([-1, 5, 10, 15, 20], size=data_size, p=[0.85, 0.05, 0.05, 0.025, 0.025]),
    'previous': np.random.randint(0, 10, size=data_size),
    'poutcome': np.random.choice(['failure', 'success', 'unknown'], size=data_size, p=[0.6, 0.1, 0.3]),
    'y': np.random.choice([0, 1], size=data_size, p=[0.88, 0.12])  # target variable: 0=no, 1=yes
})

# Data Preprocessing

# Encoding categorical variables
le = LabelEncoder()
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_columns:
    synthetic_data[col] = le.fit_transform(synthetic_data[col])

# Split data into features (X) and target (y)
X = synthetic_data.drop('y', axis=1)
y = synthetic_data['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)