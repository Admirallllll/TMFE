"""
AI Classifier Module

Fine-tunes FinBERT for AI topic classification using transfer learning.
"""

import os
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import numpy as np
from tqdm import tqdm
import json


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str = "yiyanghkust/finbert-tone"  # FinBERT
    num_labels: int = 2
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_dir: str = "outputs/models"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AIClassifier:
    """
    FinBERT-based classifier for AI topic detection.
    
    Uses transfer learning: pre-trained on financial text (FinBERT),
    fine-tuned on AI news articles.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels
        )
        self.model.to(self.device)
        
        self.training_history = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        scheduler
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of class 1
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_best: bool = True
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_best: Whether to save best model checkpoint
            
        Returns:
            Training history
        """
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        best_val_f1 = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print('='*50)
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler)
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            
            # Save best model
            if save_best and val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save(os.path.join(self.config.save_dir, 'best_model'))
                print(f"  >> Saved new best model (F1: {best_val_f1:.4f})")
        
        # Save final model
        self.save(os.path.join(self.config.save_dir, 'final_model'))
        
        # Save training history
        with open(os.path.join(self.config.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def predict(
        self,
        texts: list,
        batch_size: int = 32,
        return_probs: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict on new texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for inference
            return_probs: Whether to return probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i+batch_size]
            
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            if return_probs:
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        if return_probs:
            return np.array(all_preds), np.array(all_probs)
        return np.array(all_preds), None
    
    def save(self, path: str):
        """Save model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config_dict = {k: v for k, v in vars(self.config).items()}
        with open(os.path.join(path, 'training_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None) -> 'AIClassifier':
        """
        Load saved model.
        
        Args:
            path: Path to saved model directory
            device: Device to load model to
            
        Returns:
            Loaded AIClassifier instance
        """
        config_path = os.path.join(path, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = TrainingConfig(**config_dict)
        else:
            config = TrainingConfig()
        
        if device:
            config.device = device
        
        # Create instance with dummy model name (will be overwritten)
        instance = cls.__new__(cls)
        instance.config = config
        instance.device = torch.device(config.device)
        instance.training_history = []
        
        # Load tokenizer and model from path
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.model.to(instance.device)
        
        print(f"Model loaded from {path}")
        return instance


def train_classifier(
    train_data_path: str,
    val_data_path: str,
    output_dir: str = "outputs/models",
    epochs: int = 3,
    batch_size: int = 16,
    sample_n: Optional[int] = None
):
    """
    Full training pipeline for AI classifier.
    
    Args:
        train_data_path: Path to training data parquet
        val_data_path: Path to validation data parquet
        output_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Batch size
        sample_n: Number of samples for testing
    """
    import pandas as pd
    from .ai_news_dataset import AINewsDataset
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_parquet(train_data_path)
    val_df = pd.read_parquet(val_data_path)
    
    if sample_n:
        train_df = train_df.head(sample_n)
        val_df = val_df.head(sample_n // 5)
        print(f"Using sample of {len(train_df)} train, {len(val_df)} val")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Initialize classifier
    config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        save_dir=output_dir
    )
    classifier = AIClassifier(config)
    
    # Create datasets
    train_dataset = AINewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=classifier.tokenizer,
        max_length=config.max_length
    )
    
    val_dataset = AINewsDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=classifier.tokenizer,
        max_length=config.max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train
    history = classifier.train(train_loader, val_loader)
    
    print("\n=== Training Complete ===")
    print(f"Best Val F1: {max(history['val_f1']):.4f}")
    
    return classifier, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AI classifier")
    parser.add_argument("--train-data", default="outputs/features/ai_news_train.parquet")
    parser.add_argument("--val-data", default="outputs/features/ai_news_val.parquet")
    parser.add_argument("--output-dir", default="outputs/models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample", type=int, default=None, help="Sample size for dev mode")
    parser.add_argument("--dev", action="store_true", help="Development mode with small sample")
    
    args = parser.parse_args()
    
    if args.dev:
        args.sample = args.sample or 100
        args.epochs = 1
    
    train_classifier(
        args.train_data,
        args.val_data,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.sample
    )
