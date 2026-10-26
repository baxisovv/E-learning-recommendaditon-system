import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self):
        """Loads all MARS datasets and standardizes column schemas."""
        users = pd.read_csv(os.path.join(self.data_dir, 'users.csv'), encoding='utf-8')
        items = pd.read_csv(os.path.join(self.data_dir, 'items.csv'), encoding='utf-8')
        explicit = pd.read_csv(os.path.join(self.data_dir, 'explicit_ratings.csv'), encoding='utf-8')
        implicit = pd.read_csv(os.path.join(self.data_dir, 'implicit_ratings.csv'), encoding='utf-8')

        # 1. Standardize Users (ID -> user_id, Job -> job)
        users.columns = [col.strip().lower() for col in users.columns]
        users.rename(columns={'id': 'user_id'}, inplace=True)

        # 2. Standardize Items (ID -> item_id, Name -> name, etc.)
        items.columns = [col.strip().lower() for col in items.columns]
        items.rename(columns={'id': 'item_id'}, inplace=True)

        # 3. Standardize Explicit Ratings (Watch% -> watch_percent, etc.)
        # We also enforce user_id and item_id just in case the CSV uses 'UserID' or 'User ID'
        explicit.columns = [col.strip().lower().replace('%', '_percent').replace(' ', '_') for col in explicit.columns]
        explicit.rename(columns={'userid': 'user_id', 'itemid': 'item_id'}, inplace=True)

        # 4. Standardize Implicit Ratings
        implicit.columns = [col.strip().lower().replace(' ', '_') for col in implicit.columns]
        implicit.rename(columns={'userid': 'user_id', 'itemid': 'item_id'}, inplace=True)

        return users, items, explicit, implicit