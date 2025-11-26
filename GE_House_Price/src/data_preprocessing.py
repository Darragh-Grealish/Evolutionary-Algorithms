import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    # drop missing values
    df = df.dropna()
    # encode categorical features
    df['city_num'] = df['city'].astype('category').cat.codes
    df['statezip_num'] = df['statezip'].astype('category').cat.codes
    df['country_num'] = df['country'].astype('category').cat.codes

    features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
        'yr_built', 'yr_renovated', 'city_num', 'statezip_num', 'country_num'
    ]
    X = df[features].values
    y = df['price'].values
    return train_test_split(X, y, test_size=0.2, random_state=1)
