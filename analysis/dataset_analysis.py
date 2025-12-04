import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("./data/houses.csv")

# Select numeric columns that could affect price
numeric_features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated'
]

# Compute correlation with price
corr_with_price = df[numeric_features + ['price']].corr()['price'].drop('price')

# Plot
plt.figure(figsize=(10,6))
corr_with_price.sort_values().plot(kind='barh', color='skyblue')
plt.title("Correlation of Features with Price")
plt.xlabel("Correlation coefficient")
plt.ylabel("Feature")
plt.show()
