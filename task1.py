import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate sample data
np.random.seed(42)
ages = np.random.randint(18, 80, size=500)  # Generate 500 random ages between 18 and 80
genders = np.random.choice(['Male', 'Female', 'Other'], size=500, p=[0.48, 0.48, 0.04])  # Gender distribution

# Create a DataFrame
data = pd.DataFrame({'Age': ages, 'Gender': genders})

# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Histogram for age distribution
ax[0].hist(data['Age'], bins=10, color='skyblue', edgecolor='black')
ax[0].set_title('Age Distribution')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Frequency')

# Bar chart for gender distribution
gender_counts = data['Gender'].value_counts()
ax[1].bar(gender_counts.index, gender_counts.values, color=['lightcoral', 'lightblue', 'lightgreen'])
ax[1].set_title('Gender Distribution')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('Count')

# Display the plots
plt.tight_layout()
plt.show()