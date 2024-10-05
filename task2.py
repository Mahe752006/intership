import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generating a synthetic Titanic-like dataset
np.random.seed(42)

# Sample synthetic Titanic dataset (simulated)
titanic_data = pd.DataFrame({
    'PassengerId': np.arange(1, 501),
    'Survived': np.random.choice([0, 1], size=500, p=[0.62, 0.38]),  # survival probability
    'Pclass': np.random.choice([1, 2, 3], size=500, p=[0.24, 0.18, 0.58]),  # class distribution
    'Name': ['Passenger' + str(i) for i in range(1, 501)],
    'Sex': np.random.choice(['male', 'female'], size=500, p=[0.5, 0.5]),
    'Age': np.random.normal(30, 14, 500).round(1),  # Normal distribution for age with mean 30
    'SibSp': np.random.randint(0, 6, size=500),  # Number of siblings/spouses aboard
    'Parch': np.random.randint(0, 6, size=500),  # Number of parents/children aboard
    'Ticket': ['Ticket' + str(i) for i in range(1, 501)],
    'Fare': np.random.normal(50, 15, 500).round(2),  # Fare price
    'Embarked': np.random.choice(['C', 'Q', 'S'], size=500, p=[0.2, 0.1, 0.7])  # Embarkation port
})

# Introduce some missing values
titanic_data.loc[np.random.choice(titanic_data.index, size=50), 'Age'] = np.nan  # Missing ages
titanic_data.loc[np.random.choice(titanic_data.index, size=10), 'Embarked'] = np.nan  # Missing embarked

# Data Cleaning

# 1. Handling missing values
# Fill missing age values with the median
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Fill missing embarked values with the most common port
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# 2. Correcting data types (none needed in this synthetic dataset)

# Exploratory Data Analysis (EDA)

# Summary statistics
summary_stats = titanic_data.describe()

# Visualizing survival rates by class and gender
survival_by_class_gender = titanic_data.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()

# Visualizing distributions of Age, Fare, and Survived
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Distribution of ages
ax[0, 0].hist(titanic_data['Age'], bins=15, color='skyblue', edgecolor='black')
ax[0, 0].set_title('Age Distribution')
ax[0, 0].set_xlabel('Age')
ax[0, 0].set_ylabel('Frequency')

# Survival rate by Pclass and Gender
survival_by_class_gender.plot(kind='bar', stacked=False, ax=ax[0, 1], color=['lightcoral', 'lightblue'])
ax[0, 1].set_title('Survival Rate by Class and Gender')
ax[0, 1].set_ylabel('Survival Rate')

# Fare distribution
ax[1, 0].hist(titanic_data['Fare'], bins=20, color='lightgreen', edgecolor='black')
ax[1, 0].set_title('Fare Distribution')
ax[1, 0].set_xlabel('Fare')
ax[1, 0].set_ylabel('Frequency')

# Survival counts
titanic_data['Survived'].value_counts().plot(kind='bar', ax=ax[1, 1], color=['lightcoral', 'lightblue'])
ax[1, 1].set_title('Survival Count')
ax[1, 1].set_xlabel('Survived (0 = No, 1 = Yes)')
ax[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# Display the summary statistics
print(summary_stats)