"""
AI News Dataset Module

Loads and preprocesses the AI media dataset for training the transfer learning model.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import ast


@dataclass
class DataConfig:
    """Configuration for dataset loading."""
    max_length: int = 256
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_content_length: int = 50
    random_state: int = 42


class AINewsDataset(Dataset):
    """
    PyTorch Dataset for AI news classification.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256
    ):
        """
        Args:
            texts: List of text strings
            labels: List of binary labels (1 = AI-related, 0 = not)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class AINewsDataLoader:
    """
    Loader for the AI media dataset with preprocessing and splitting.
    """
    
    # Keywords that indicate AI-related content (for labeling)
    AI_TAGS = [
        'AI', 'Artificial Intelligence', 'Machine Learning', 'Deep Learning',
        'NLP', 'Computer Vision', 'Robotics', 'Automation', 'Neural Network',
        'ChatGPT', 'GPT', 'LLM', 'Generative AI', 'GenAI', 'OpenAI',
        'Data Science', 'Predictive Analytics', 'Natural Language Processing'
    ]
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Load AI news CSV file.
        
        Args:
            path: Path to CSV file
            
        Returns:
            DataFrame with news data
        """
        print(f"Loading AI news data from {path}...")
        df = pd.read_csv(path, low_memory=False)
        
        print(f"Loaded {len(df)} articles")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def parse_tags(self, tags_str) -> List[str]:
        """Parse tags field which may be a string representation of a list."""
        if pd.isna(tags_str):
            return []
        
        if isinstance(tags_str, list):
            return tags_str
        
        try:
            # Try literal eval for string like "['tag1', 'tag2']"
            return ast.literal_eval(str(tags_str))
        except:
            # Fall back to string parsing
            tags_str = str(tags_str)
            tags_str = tags_str.strip('[]')
            tags = [t.strip().strip("'\"") for t in tags_str.split(',')]
            return [t for t in tags if t]
    
    def is_ai_related(self, tags: List[str], content: str = "") -> bool:
        """
        Determine if article is AI-related based on tags and content.
        
        Args:
            tags: List of article tags
            content: Article content (optional)
            
        Returns:
            True if AI-related
        """
        # Check tags
        tags_lower = [t.lower() for t in tags]
        for ai_tag in self.AI_TAGS:
            if ai_tag.lower() in tags_lower:
                return True
            # Partial match
            for t in tags_lower:
                if ai_tag.lower() in t:
                    return True
        
        return False
    
    def preprocess(
        self, 
        df: pd.DataFrame,
        content_col: str = 'content',
        tags_col: str = 'tags',
        title_col: str = 'title'
    ) -> pd.DataFrame:
        """
        Preprocess the dataset: clean text and create labels.
        
        Args:
            df: Raw DataFrame
            content_col: Column with article content
            tags_col: Column with tags
            title_col: Column with title
            
        Returns:
            Preprocessed DataFrame
        """
        print("Preprocessing dataset...")
        
        # Copy to avoid modifying original
        df = df.copy()
        
        # Combine title and content
        df['text'] = df.apply(
            lambda x: f"{str(x.get(title_col, ''))} {str(x.get(content_col, ''))}",
            axis=1
        )
        
        # Clean text
        df['text'] = df['text'].apply(self._clean_text)
        
        # Filter by length
        df['text_length'] = df['text'].str.len()
        df = df[df['text_length'] >= self.config.min_content_length].copy()
        print(f"After filtering by length: {len(df)} articles")
        
        # Parse tags and create labels
        df['parsed_tags'] = df[tags_col].apply(self.parse_tags)
        df['is_ai'] = df.apply(
            lambda x: self.is_ai_related(x['parsed_tags'], x['text']),
            axis=1
        )
        df['label'] = df['is_ai'].astype(int)
        
        print(f"AI-related articles: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
        print(f"Non-AI articles: {(~df['is_ai']).sum()} ({(~df['is_ai']).mean()*100:.1f}%)")
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean article text."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        return text.strip()
    
    def create_splits(
        self,
        df: pd.DataFrame,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets.
        
        Args:
            df: Preprocessed DataFrame
            stratify: Whether to stratify by label
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("Creating train/val/test splits...")
        
        stratify_col = df['label'] if stratify else None
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=self.config.train_ratio,
            random_state=self.config.random_state,
            stratify=stratify_col
        )
        
        # Second split: val vs test
        val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            random_state=self.config.random_state,
            stratify=temp_df['label'] if stratify else None
        )
        
        print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def create_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer,
        batch_size: int = 16,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            train_df, val_df, test_df: Split DataFrames
            tokenizer: HuggingFace tokenizer
            batch_size: Batch size
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset = AINewsDataset(
            texts=train_df['text'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=self.config.max_length
        )
        
        val_dataset = AINewsDataset(
            texts=val_df['text'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=self.config.max_length
        )
        
        test_dataset = AINewsDataset(
            texts=test_df['text'].tolist(),
            labels=test_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=self.config.max_length
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader


def prepare_training_data(
    csv_path: str,
    output_dir: str = "outputs/features",
    sample_n: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Full pipeline to prepare training data from AI news CSV.
    
    Args:
        csv_path: Path to AI news CSV
        output_dir: Directory to save processed data
        sample_n: Number of samples for testing
        
    Returns:
        Dictionary with train/val/test DataFrames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    loader = AINewsDataLoader()
    
    # Load and preprocess
    df = loader.load_csv(csv_path)
    
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
        print(f"Sampled {len(df)} articles for testing")
    
    df = loader.preprocess(df)
    
    # Create splits
    train_df, val_df, test_df = loader.create_splits(df)
    
    # Save processed data
    train_df.to_parquet(f"{output_dir}/ai_news_train.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/ai_news_val.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/ai_news_test.parquet", index=False)
    
    print(f"\nSaved processed data to {output_dir}/")
    
    return {'train': train_df, 'val': val_df, 'test': test_df}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare AI news training data")
    parser.add_argument("--input", default="ai_media_dataset_20250911.csv")
    parser.add_argument("--output-dir", default="outputs/features")
    parser.add_argument("--sample", type=int, default=None)
    
    args = parser.parse_args()
    
    prepare_training_data(args.input, args.output_dir, args.sample)
