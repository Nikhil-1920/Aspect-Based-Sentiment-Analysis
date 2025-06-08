import sys
import os
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW  # Import from torch.optim instead of transformers
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import load
import json
import nltk
from tqdm import tqdm
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nltk.download('sentiwordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Could not download NLTK resources. If needed, please install manually.")

try:
    nlp = spacy.load('en_core_web_sm')
except:
    logger.warning("Could not load spaCy model. Please install with: python -m spacy download en_core_web_sm")

class HybridABSADataset(Dataset):
    """Dataset for hybrid aspect-based sentiment analysis combining BERT with traditional features"""
    
    def __init__(self, dataframe, tokenizer, tfidf_vectorizer=None, max_length=128):
        """
        Args:
            dataframe: Pandas dataframe with 'sentence', 'aspect', and 'polarity' columns
            tokenizer: BERT tokenizer
            tfidf_vectorizer: Trained TF-IDF vectorizer
            max_length: Maximum sequence length
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.tfidf_vectorizer = tfidf_vectorizer
        self.max_length = max_length
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sentence = self.dataframe.iloc[idx]['sentence']
        aspect = self.dataframe.iloc[idx]['aspect']
        
        input_text = f"{sentence} [SEP] {aspect}"
        
        # Get BERT encodings
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
        
        if 'lexicon_pos' in self.dataframe.columns and 'lexicon_neg' in self.dataframe.columns:
            lexicon_features = torch.tensor([
                self.dataframe.iloc[idx]['lexicon_pos'],
                self.dataframe.iloc[idx]['lexicon_neg']
            ], dtype=torch.float)
            item['lexicon_features'] = lexicon_features
        
        if 'polarity' in self.dataframe.columns:
            polarity = self.dataframe.iloc[idx]['polarity']
            item['labels'] = torch.tensor(self.label_map[polarity])

        tfidf_cols = [col for col in self.dataframe.columns if col.startswith('tfidf_')]
        if tfidf_cols:
            tfidf_values = self.dataframe.iloc[idx][tfidf_cols].values.astype(np.float32)
            item['tfidf_features'] = torch.tensor(tfidf_values, dtype=torch.float)
        
        return item    
            
def get_lexicon_features(text):
    """Extract sentiment lexicon scores using SentiWordNet"""
    pos_tags = pos_tag(word_tokenize(text))
    pos_scores = []
    neg_scores = []
    
    for word, tag in pos_tags:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            pos_scores.append(synsets[0].pos_score())
            neg_scores.append(synsets[0].neg_score())
    
    return {
        'lexicon_pos': np.mean(pos_scores) if pos_scores else 0,
        'lexicon_neg': np.mean(neg_scores) if neg_scores else 0
    }

class HybridBERTModel(nn.Module):
    """Hybrid BERT model for aspect-based sentiment analysis"""
    
    def __init__(self, bert_model_name="bert-base-uncased", num_labels=3, 
                 tfidf_dim=None, lexicon_dim=2, dropout_rate=0.1):
        """
        Args:
            bert_model_name: Name of the BERT model to use
            num_labels: Number of sentiment labels (3 for positive, negative, neutral)
            tfidf_dim: Dimension of TF-IDF features (None if not used)
            lexicon_dim: Dimension of lexicon features (2 for pos/neg)
            dropout_rate: Dropout rate for regularization
        """
        super(HybridBERTModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size  # 768 for bert-base
        
        self.tfidf_dim = tfidf_dim
        self.lexicon_dim = lexicon_dim
        
        # Attention layer for feature weighting
        self.feature_attention = None
        self.feature_attention = nn.Sequential(
            nn.Linear(self.bert_dim + (self.tfidf_dim or 0) + self.lexicon_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 3 if self.tfidf_dim else 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim + (self.tfidf_dim or 0) + self.lexicon_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(512),
            nn.Linear(512, num_labels)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask, token_type_ids, 
                tfidf_features=None, lexicon_features=None):
        """
        Forward pass with optional TF-IDF and lexicon features
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] embedding as sentence representation
        bert_embedding = outputs.last_hidden_state[:, 0, :]
        
        if tfidf_features is not None and lexicon_features is not None:
            combined_features = torch.cat([bert_embedding, tfidf_features, lexicon_features], dim=1)
            
            # Apply attention weighting
            attention_weights = self.softmax(self.feature_attention(combined_features))
            
            bert_weight = attention_weights[:, 0].unsqueeze(1)
            tfidf_weight = attention_weights[:, 1].unsqueeze(1)
            lexicon_weight = attention_weights[:, 2].unsqueeze(1)
            
            weighted_bert = bert_embedding * bert_weight.expand_as(bert_embedding)
            weighted_tfidf = tfidf_features * tfidf_weight.expand_as(tfidf_features)
            weighted_lexicon = lexicon_features * lexicon_weight.expand_as(lexicon_features)
            
            weighted_features = torch.cat([weighted_bert, weighted_tfidf, weighted_lexicon], dim=1)
        
        elif lexicon_features is not None:
            # Concatenate BERT and lexicon features
            combined_features = torch.cat([bert_embedding, lexicon_features], dim=1)
            
            attention_weights = self.softmax(self.feature_attention(combined_features))
            
            bert_weight = attention_weights[:, 0].unsqueeze(1)
            lexicon_weight = attention_weights[:, 1].unsqueeze(1)
            
            weighted_bert = bert_embedding * bert_weight.expand_as(bert_embedding)
            weighted_lexicon = lexicon_features * lexicon_weight.expand_as(lexicon_features)
            
            # Concatenate weighted features
            weighted_features = torch.cat([weighted_bert, weighted_lexicon], dim=1)
        
        else:
            weighted_features = bert_embedding
        
        logits = self.classifier(weighted_features)
        
        return logits

class HybridABSA_Classifier:
    """Hybrid BERT classifier for Aspect-Based Sentiment Analysis"""
    
    def __init__(self, model_name="bert-base-uncased", num_labels=3, 
                 tfidf_dim=None, tfidf_vectorizer=None,
                 lexicon_dim=2, use_attention=True):
        """
        Args:
            model_name: Name of the BERT model to use
            num_labels: Number of sentiment labels (3 for positive, negative, neutral)
            tfidf_dim: Dimension of TF-IDF features after dimensionality reduction
            tfidf_vectorizer: Pre-trained TF-IDF vectorizer
            lexicon_dim: Dimension of lexicon features
            use_attention: Whether to use attention mechanism for feature weighting
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # TF-IDF settings
        self.tfidf_dim = tfidf_dim
        self.tfidf_vectorizer = tfidf_vectorizer
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = HybridBERTModel(
            bert_model_name=model_name,
            num_labels=num_labels,
            tfidf_dim=tfidf_dim,
            lexicon_dim=lexicon_dim
        )
        self.model.to(self.device)
        
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def prepare_traditional_features(self, df):
        """Prepare traditional features (TF-IDF and lexicon)"""
        df['text'] = df['sentence'] + " " + df['aspect']
        
        # Add lexicon features if not present
        if 'lexicon_pos' not in df.columns or 'lexicon_neg' not in df.columns:
            logger.info("Extracting lexicon features...")
            lexicon_features = []
            for text in tqdm(df['sentence']):
                lexicon_features.append(get_lexicon_features(text))
            
            lexicon_df = pd.DataFrame(lexicon_features)
            df = pd.concat([df, lexicon_df], axis=1)
        
        # Process TF-IDF features if vectorizer is provided
        if self.tfidf_vectorizer is not None:
            logger.info("Extracting TF-IDF features...")
            tfidf_features = self.tfidf_vectorizer.transform(df['text']).toarray()
            
            if self.tfidf_dim is not None and self.tfidf_dim < tfidf_features.shape[1]:
                from sklearn.decomposition import TruncatedSVD
                logger.info(f"Reducing TF-IDF dimensions from {tfidf_features.shape[1]} to {self.tfidf_dim}...")
                svd = TruncatedSVD(n_components=self.tfidf_dim)
                tfidf_features = svd.fit_transform(tfidf_features)
            
            # Add TF-IDF features to dataframe
            tfidf_df = pd.DataFrame(
                tfidf_features, 
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])],
                index=df.index  
            )

            df = pd.concat([df, tfidf_df], axis=1)
        
        return df
        
    def prepare_dataloader(self, df, batch_size=16, shuffle=True):
        """Create DataLoader from dataframe"""
        df = self.prepare_traditional_features(df)
        
        dataset = HybridABSADataset(df, self.tokenizer, self.tfidf_vectorizer)
        
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
            
            lexicon_features = torch.stack([item['lexicon_features'] for item in batch]) if 'lexicon_features' in batch[0] else None
            
            tfidf_features = torch.stack([item['tfidf_features'] for item in batch]) if 'tfidf_features' in batch[0] else None
            
            labels = torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'tfidf_features': tfidf_features,
                'lexicon_features': lexicon_features,
                'labels': labels
            }
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        
    def train(self, train_csv, val_csv, output_dir="hybrid_bert_model", 
              batch_size=16, epochs=4, learning_rate=2e-5, warmup_steps=0):
        """Train the hybrid model"""
        os.makedirs(output_dir, exist_ok=True)
        
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        # Initialize TF-IDF vectorizer if not provided
        if self.tfidf_vectorizer is None and self.tfidf_dim is not None:
            logger.info("Initializing TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=10000 if self.tfidf_dim > 10000 else None,
                stop_words='english'
            )
            train_df['text'] = train_df['sentence'] + " " + train_df['aspect']
            self.tfidf_vectorizer.fit(train_df['text'])
        
        train_dataloader = self.prepare_dataloader(train_df, batch_size=batch_size)
        val_dataloader = self.prepare_dataloader(val_df, batch_size=batch_size, shuffle=False)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        loss_fn = nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        metrics = {'training': {}, 'validation': {}}
        
        desc_validation = "Validation"
        desc_training   = "Training  " 
        for epoch in range(epochs):
            logger.info(f"====== Epoch {epoch+1}/{epochs} ======")
            
            # Training
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_dataloader, desc=desc_training):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                tfidf_features = batch['tfidf_features'].to(self.device) if batch['tfidf_features'] is not None else None
                lexicon_features = batch['lexicon_features'].to(self.device) if batch['lexicon_features'] is not None else None
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    tfidf_features=tfidf_features,
                    lexicon_features=lexicon_features
                )
                
                loss = loss_fn(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_dataloader)
            logger.info(f"Train loss: {avg_train_loss:.4f}")
            
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=desc_validation):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    tfidf_features = batch['tfidf_features'].to(self.device) if batch['tfidf_features'] is not None else None
                    lexicon_features = batch['lexicon_features'].to(self.device) if batch['lexicon_features'] is not None else None
                    
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        tfidf_features=tfidf_features,
                        lexicon_features=lexicon_features
                    )
                    
                    loss = loss_fn(logits, labels)
                    val_loss += loss.item()
                    
                    # Get predictions
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels = labels.cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            
            avg_val_loss = val_loss / len(val_dataloader)
            
            val_accuracy = accuracy_score(all_labels, all_preds)
            precision_macro = precision_score(all_labels, all_preds, average='macro')
            recall_macro = recall_score(all_labels, all_preds, average='macro')
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
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': val_f1,
                'confusion_matrix': val_confusion
            }

            precision_per_class = precision_score(all_labels, all_preds, average=None)
            recall_per_class = recall_score(all_labels, all_preds, average=None)
            metrics['validation'][f'epoch_{epoch+1}'].update({
                'precision_per_class': {k: float(v) for k, v in zip(self.reverse_label_map.values(), precision_per_class)},
                'recall_per_class': {k: float(v) for k, v in zip(self.reverse_label_map.values(), recall_per_class)}
            })
            
            # Save best model
            if val_f1 > best_val_f1:
                logger.info(f"New best model! F1: {val_f1:.4f}")
                best_val_f1 = val_f1
                
                model_path = os.path.join(output_dir, f"best_model")
                os.makedirs(model_path, exist_ok=True)
                
                torch.save(self.model.state_dict(), os.path.join(model_path, "hybrid_model.pt"))
                self.tokenizer.save_pretrained(model_path)
                
                # Save TF-IDF vectorizer
                if self.tfidf_vectorizer is not None:
                    from joblib import dump
                    dump(self.tfidf_vectorizer, os.path.join(model_path, "tfidf_vectorizer.joblib"))
                
                config = {
                    'model_name': self.model_name,
                    'tfidf_dim': self.tfidf_dim,
                    'lexicon_dim': 2,   # Fixed for now
                    'num_labels': self.num_labels
                }
                with open(os.path.join(model_path, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)
                
                # Save metrics
                with open(os.path.join(output_dir, "best_metrics.json"), "w") as f:
                    json.dump({
                        'epoch': epoch + 1,
                        'accuracy': val_accuracy,
                        'precision_macro': precision_macro,
                        'recall_macro': recall_macro,
                        'f1_macro': val_f1,
                        'confusion_matrix': val_confusion,
                        'precision_per_class': precision_per_class.tolist(),
                        'recall_per_class': recall_per_class.tolist()     
                    }, f, indent=2)
        
        with open(os.path.join(output_dir, "all_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
    
    def load_model(self, model_path):
        """Load a saved model"""
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.tfidf_dim = config.get('tfidf_dim')
        lexicon_dim = config.get('lexicon_dim', 2)
        self.num_labels = config.get('num_labels', 3)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        self.model = HybridBERTModel(
            bert_model_name=self.model_name,
            num_labels=self.num_labels,
            tfidf_dim=self.tfidf_dim,
            lexicon_dim=lexicon_dim
        )
        
        self.model.load_state_dict(torch.load(os.path.join(model_path, "hybrid_model.pt")))
        self.model.to(self.device)
        
        # Load TF-IDF vectorizer 
        if os.path.exists(os.path.join(model_path, "tfidf_vectorizer.joblib")):
            from joblib import load
            self.tfidf_vectorizer = load(os.path.join(model_path, "tfidf_vectorizer.joblib"))
        
    # def predict(self, sentences, aspects):
    #     """Predict sentiment for sentence-aspect pairs"""
    #     self.model.eval()
    #     results = []
        
    #     # Create dataframe
    #     df = pd.DataFrame({'sentence': sentences, 'aspect': aspects})
        
    #     # Prepare traditional features
    #     df = self.prepare_traditional_features(df)
        
    #     # Create dataset and dataloader
    #     dataset = HybridABSADataset(df, self.tokenizer, self.tfidf_vectorizer)
    #     dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
    #     with torch.no_grad():
    #         for batch in tqdm(dataloader, desc="Predicting"):
    #             # Move batch to device
    #             input_ids = batch['input_ids'].to(self.device)
    #             attention_mask = batch['attention_mask'].to(self.device)
    #             token_type_ids = batch['token_type_ids'].to(self.device)
                
    #             # Optional features
    #             tfidf_features = batch['tfidf_features'].to(self.device) if 'tfidf_features' in batch else None
    #             lexicon_features = batch['lexicon_features'].to(self.device) if 'lexicon_features' in batch else None
                
    #             # Forward pass
    #             logits = self.model(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 token_type_ids=token_type_ids,
    #                 tfidf_features=tfidf_features,
    #                 lexicon_features=lexicon_features
    #             )
                
    #             # Get predictions
    #             preds = torch.argmax(logits, dim=1).cpu().numpy()
                
    #             probs = torch.softmax(logits, dim=1)
            
    #             for idx, (pred, prob) in enumerate(zip(preds, probs)):
    #                 results.append({
    #                     'sentence': sentences[idx],
    #                     'aspect': aspects[idx],
    #                     'predicted_polarity': self.reverse_label_map[pred],
    #                     'confidence': prob[pred].item(),
    #                     'probabilities': {k: v.item() for k, v in zip(self.reverse_label_map.values(), prob)}
    #                 })
        
    #     return pd.DataFrame(results)

    def predict(self, sentences, aspects):
        """Predict sentiment for sentence-aspect pairs"""
        self.model.eval()
        results = []
        
        df = pd.DataFrame({'sentence': sentences, 'aspect': aspects})
        
        df['text'] = df['sentence'] + " " + df['aspect']
        if 'lexicon_pos' not in df.columns:
            df['lexicon_pos'] = 0
        if 'lexicon_neg' not in df.columns:
            df['lexicon_neg'] = 0
        
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
            lexicon_features = torch.stack([item['lexicon_features'] for item in batch]) if 'lexicon_features' in batch[0] else None
            tfidf_features = torch.stack([item['tfidf_features'] for item in batch]) if 'tfidf_features' in batch[0] else None
            labels = torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
            return {
                'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                'tfidf_features': tfidf_features, 'lexicon_features': lexicon_features, 'labels': labels
            }
        
        dataset = HybridABSADataset(df, self.tokenizer, self.tfidf_vectorizer)
        dataloader = DataLoader(dataset, batch_size=len(sentences), shuffle=False, collate_fn=collate_fn) # Use collate_fn
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                tfidf_features = batch['tfidf_features'].to(self.device) if batch['tfidf_features'] is not None else None
                lexicon_features = batch['lexicon_features'].to(self.device) if batch['lexicon_features'] is not None else None
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    tfidf_features=tfidf_features,
                    lexicon_features=lexicon_features
                )
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probs = torch.softmax(logits, dim=1).cpu()

                for i in range(len(preds)):
                    pred_class = preds[i]
                    confidence = probs[i][pred_class].item()
                    results.append({
                        'aspect': aspects[i],
                        'polarity': self.reverse_label_map[pred_class],
                        'confidence': confidence
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

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                tfidf_features = batch['tfidf_features'].to(self.device) if batch['tfidf_features'] is not None else None
                lexicon_features = batch['lexicon_features'].to(self.device) if batch['lexicon_features'] is not None else None
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    tfidf_features=tfidf_features,
                    lexicon_features=lexicon_features
                )
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        test_df['predicted_polarity'] = [self.reverse_label_map[p] for p in all_preds]
        
        if output_csv:
            test_df[['sentence', 'aspect', 'predicted_polarity']].to_csv(output_csv, index=False)
            
        return test_df