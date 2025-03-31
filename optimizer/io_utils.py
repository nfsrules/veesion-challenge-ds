# optimizer/io_utils.py
import os
import pandas as pd
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

DEFAULT_THEFT_LABELS = [
    'Suspicious Bag', 'Suspicious', 'Theft',
    'Gesture Into Body', 'Product Into Stroller'
]

class DataLoader:
    def __init__(self, source: str, source_type: str = "csv", theft_labels: Optional[List[str]] = None):
        """
        Flexible data loader.

        Args:
            source (str): File path or connection string.
            source_type (str): One of ['csv'].
            theft_labels (List[str], optional): Override default labels.
        """
        self.source = source
        self.source_type = source_type
        self.theft_labels = theft_labels or DEFAULT_THEFT_LABELS

    def load(self) -> pd.DataFrame:
        if self.source_type == "csv":
            return self._load_csv()
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}")

    def _load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"File not found: {self.source}")
        
        logger.info(f"Loading CSV from {self.source}")
        df = pd.read_csv(self.source, delimiter=";", index_col=0)

        if 'label' not in df.columns or 'video_name' not in df.columns:
            raise ValueError("Missing required columns: 'label' and/or 'video_name'")

        df.dropna(inplace=True)
        df['is_theft'] = df['label'].isin(self.theft_labels).astype(int)
        df['camera_id'] = df['video_name'].str.extract(r'camera_(\d+)_ip')

        if df['camera_id'].isnull().any():
            logger.warning("Some rows had missing or unparsable camera_id values.")

        return df
