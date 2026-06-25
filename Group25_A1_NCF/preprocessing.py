import pandas as pd
import os
import requests
import zipfile
from io import BytesIO
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'user_id': torch.tensor(row['user_id'], dtype=torch.long),
            'movie_id': torch.tensor(row['movie_id'], dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.float)
        }


def download_dataset():
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    print("Downloading MovieLens 1M dataset...")
    response = requests.get(url)

    if response.status_code == 200:
        z = zipfile.ZipFile(BytesIO(response.content))
        z.extractall("data/")
        print("Dataset downloaded and extracted successfully")
        return "data/ml-1m"
    else:
        raise Exception("Failed to download dataset")


def load_dataset(data_path="data/ml-1m"):
    if not os.path.exists(data_path):
        data_path = download_dataset()

    # Load ratings data
    ratings_path = os.path.join(data_path, 'ratings.dat')
    ratings_df = pd.read_csv(
        ratings_path,
        sep='::',
        header=None,
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )

    # Convert IDs to integers (0-indexed for embedding lookup)
    user_ids = ratings_df['user_id'].unique()
    movie_ids = ratings_df['movie_id'].unique()

    user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    movie_id_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}

    ratings_df['user_id'] = ratings_df['user_id'].map(user_id_map)
    ratings_df['movie_id'] = ratings_df['movie_id'].map(movie_id_map)

    # Keep mapping for later use
    id_mappings = {
        'user_id_map': user_id_map,
        'movie_id_map': movie_id_map,
        'n_users': len(user_ids),
        'n_items': len(movie_ids)
    }

    return ratings_df, id_mappings


def preprocess_ratings(ratings_df, threshold=4.0):
    # Filter ratings >= threshold as positive interactions
    positive_df = ratings_df[ratings_df['rating'] >= threshold].copy()
    positive_df['label'] = 1

    # Select relevant columns for the model
    positive_df = positive_df[['user_id', 'movie_id', 'label']]

    return positive_df


def create_user_item_matrix(ratings_df, n_users, n_items):
    user_item_matrix = np.zeros((n_users, n_items), dtype=np.bool_)

    for _, row in ratings_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        user_item_matrix[user_id, movie_id] = True

    return user_item_matrix


def negative_sampling(positive_df, user_item_matrix, neg_ratio=1):
    neg_samples = []

    # Group by user to sample for each user
    for user_id, group in positive_df.groupby('user_id'):
        # Number of negative samples to generate for this user
        n_pos = len(group)
        n_neg = n_pos * neg_ratio

        # Get all items the user has not interacted with
        non_interacted_items = np.where(user_item_matrix[user_id] == False)[0]

        # If there are fewer non-interacted items than needed, sample with replacement
        if len(non_interacted_items) < n_neg:
            neg_item_ids = np.random.choice(non_interacted_items, size=n_neg, replace=True)
        else:
            neg_item_ids = np.random.choice(non_interacted_items, size=n_neg, replace=False)

        # Create negative samples
        for item_id in neg_item_ids:
            neg_samples.append({
                'user_id': user_id,
                'movie_id': item_id,
                'label': 0  # Negative interaction
            })

    negative_df = pd.DataFrame(neg_samples)

    return negative_df


def split_data(df, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    # First split: train + validation and test
    train_valid_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=42, stratify=df['label']
    )

    # Second split: train and validation
    valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)
    train_df, valid_df = train_test_split(
        train_valid_df, test_size=valid_ratio_adjusted,
        random_state=42, stratify=train_valid_df['label']
    )

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(valid_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    return train_df, valid_df, test_df


def preprocessing(neg_ratio=4, ):
    print("Loading and preprocessing data...")
    ratings_df, id_mappings = load_dataset()
    n_users = id_mappings['n_users']
    n_items = id_mappings['n_items']

    # Convert to binary interactions
    positive_df = preprocess_ratings(ratings_df)

    # Create user-item matrix for negative sampling
    user_item_matrix = create_user_item_matrix(positive_df, n_users, n_items)

    # Generate negative samples
    negative_df = negative_sampling(positive_df, user_item_matrix, neg_ratio=neg_ratio)

    # Combine positive and negative samples
    full_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Split data
    train_df, valid_df, test_df = split_data(full_df)

    train_dataset = NCFDataset(train_df)
    valid_dataset = NCFDataset(valid_df)
    test_dataset = NCFDataset(test_df)

    preprocessed_data = (train_dataset, valid_dataset, test_dataset, n_users, n_items, id_mappings)
    return preprocessed_data
