import sys
import os
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # AdamW is now in torch.optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from tqdm import tqdm
import logging
from transformers import logging as hf_logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()

from transformers import logging as hf_logging
class AspectSentimentDataset(Dataset):
    """Dataset for aspect-based sentiment analysis using BERT"""
    
    def __init__(self, dataframe, tokenizer, max_length=128):
        """
        Args:
            dataframe: Pandas dataframe with 'sentence', 'aspect', and 'polarity' columns
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sentence = self.dataframe.iloc[idx]['sentence']
        aspect = self.dataframe.iloc[idx]['aspect']
        
        # Format input as [CLS] sentence [SEP] aspect [SEP]
        input_text = f"{sentence} [SEP] {aspect}"
        
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }
        
        if 'polarity' in self.dataframe.columns:
            polarity = self.dataframe.iloc[idx]['polarity']
            item['labels'] = torch.tensor(self.label_map[polarity])
            
        return item

class ABSA_BERT_Classifier:
    """BERT-based classifier for Aspect-Based Sentiment Analysis"""
    
    def __init__(self, model_name="bert-base-uncased", num_labels=3):
        """
        Args:
            model_name: Name of the BERT model to use
            num_labels: Number of sentiment labels (3 for positive, negative, neutral)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def prepare_dataloader(self, df, batch_size=16, shuffle=True):
        """Create DataLoader from dataframe"""
        dataset = AspectSentimentDataset(df, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
    def train(self, train_csv, val_csv, output_dir="bert_model", 
              batch_size=16, epochs=4, learning_rate=2e-5, warmup_steps=0):
        """Train the BERT model"""
        os.makedirs(output_dir, exist_ok=True)
        
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        train_dataloader = self.prepare_dataloader(train_df, batch_size=batch_size)
        val_dataloader = self.prepare_dataloader(val_df, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0
        metrics = {'training': {}, 'validation': {}}
        
        for epoch in range(epochs):
            logger.info(f"====== Epoch {epoch+1}/{epochs} ======")
            
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_dataloader, desc="Training"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_dataloader)
            logger.info(f"Train loss: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):

                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            
            avg_val_loss = val_loss / len(val_dataloader)
            
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, average='macro')
            val_recall = recall_score(all_labels, all_preds, average='macro')
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            val_confusion = confusion_matrix(all_labels, all_preds).tolist()
            
            logger.info(f"Validation loss: {avg_val_loss:.4f}")
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
            logger.info(f"Validation F1 (macro): {val_f1:.4f}")
            
            # Save metrics
            metrics['training'][f'epoch_{epoch+1}'] = {'loss': avg_train_loss}
            metrics['validation'][f'epoch_{epoch+1}'] = {
                'loss': avg_val_loss,
                'accuracy': val_accuracy,
                'precision_macro': val_precision,
                'recall_macro': val_recall,
                'f1_macro': val_f1,
                'confusion_matrix': val_confusion
            }
            
            # Save best model
            if val_f1 > best_val_f1:
                logger.info(f"New best model! F1: {val_f1:.4f}")
                best_val_f1 = val_f1
                
                model_path = os.path.join(output_dir, f"best_model")
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                
                with open(os.path.join(output_dir, "best_metrics.json"), "w") as f:
                    json.dump({
                        'epoch': epoch + 1,
                        'accuracy': val_accuracy,
                        'precision_macro': val_precision,
                        'recall_macro': val_recall,
                        'f1_macro': val_f1,
                        'confusion_matrix': val_confusion
                    }, f, indent=2)
        
        with open(os.path.join(output_dir, "all_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
    
    def load_model(self, model_path):
        """Load a saved model"""
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
    def predict(self, sentences, aspects):
        """Predict sentiment for sentence-aspect pairs"""
        self.model.eval()
        results = []
        
        for sentence, aspect in zip(sentences, aspects):

            input_text = f"{sentence} [SEP] {aspect}"
            encoding = self.tokenizer.encode_plus(
                input_text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            token_type_ids = encoding['token_type_ids'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=1).item()
            polarity = self.reverse_label_map[pred_class]
            
            results.append({
                'sentence': sentence,
                'aspect': aspect,
                'predicted_polarity': polarity
            })
            
        return pd.DataFrame(results)
        
    def predict_from_test_data(self, test_csv, output_csv=None):
        """Predict polarities for preprocessed test data"""
        test_df = pd.read_csv(test_csv)
        test_dataloader = self.prepare_dataloader(test_df, batch_size=16, shuffle=False)
        
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                
                outputs = self.model(**batch)
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        # Add predictions to dataframe
        test_df['predicted_polarity'] = [self.reverse_label_map[p] for p in all_preds]
        
        # Save predictions if output path is provided
        if output_csv:
            test_df[['sentence', 'aspect', 'predicted_polarity']].to_csv(output_csv, index=False)
            
        return test_df

if __name__ == "__main__":
    domains = ['laptop', 'restaurant']
    
    for domain in domains:
        logger.info(f"Training {domain} BERT model...")
        
        output_dir = f"{domain}_bert_model"
        bert_classifier = ABSA_BERT_Classifier()
        
        # Train model
        metrics = bert_classifier.train(
            train_csv=f"{domain}_train_features.csv",
            val_csv=f"{domain}_val_features.csv",
            output_dir=output_dir,
            batch_size=16,
            epochs=4,
            learning_rate=2e-5
        )
        
        logger.info(f"{domain} training complete. Metrics saved to {output_dir}")
        
        # Test on unlabeled data
        logger.info(f"Testing {domain} model...")
        bert_classifier.load_model(f"{output_dir}/best_model")
        
        predictions = bert_classifier.predict_from_test_data(
            test_csv=f"{domain}_test_features_unlabeled.csv",
            output_csv=f"{domain}_bert_test_predictions.csv"
        )
        
        logger.info(f"{domain} predictions saved to {domain}_bert_test_predictions.csv")

