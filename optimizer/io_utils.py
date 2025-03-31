import os
import pandas as pd


THEFT_LABELS = [
    'Suspicious Bag', 'Suspicious', 'Theft',
    'Gesture Into Body', 'Product Into Stroller'
]


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Loads, cleans, and annotates the production alerts metadata CSV.

    - Reads CSV with ';' delimiter
    - Drops NaNs
    - Creates 'is_theft' column based on label list
    - Extracts 'camera_id' from 'video_name'

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned and processed DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, delimiter=";", index_col=0)
    df.dropna(inplace=True)

    df['is_theft'] = df['label'].isin(THEFT_LABELS).astype(int)
    df['camera_id'] = df['video_name'].str.extract(r'camera_(\d+)_ip')

    return df
