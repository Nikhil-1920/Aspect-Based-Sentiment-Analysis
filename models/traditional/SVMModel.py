# SVMModel.py
import sys
import os
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
import json
import os
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix)
from joblib import dump, load

class SVM_Classifier:
    def __init__(self, domain):
        self.domain = domain
        
        # Try multiple possible locations for the vectorizer files
        tfidf_paths = [
            f"{domain}_tfidf_vectorizer.joblib",
            f"data/processed/{domain}_tfidf_vectorizer.joblib", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       f"data/processed/{domain}_tfidf_vectorizer.joblib")
        ]
        
        bow_paths = [
            f"{domain}_bow_vectorizer.joblib",
            f"data/processed/{domain}_bow_vectorizer.joblib",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       f"data/processed/{domain}_bow_vectorizer.joblib")
        ]
        
        for path in tfidf_paths:
            try:
                print(f"Trying to load TF-IDF vectorizer from: {path}")
                self.tfidf = load(path)
                print(f"Successfully loaded TF-IDF vectorizer from {path}")
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(f"Could not find TF-IDF vectorizer for {domain} in any location")
        
        for path in bow_paths:
            try:
                print(f"Trying to load BOW vectorizer from: {path}")
                self.bow = load(path)
                print(f"Successfully loaded BOW vectorizer from {path}")
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(f"Could not find BOW vectorizer for {domain} in any location")
            
        self.clf = None
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def prepare_features(self, df):
        """Prepare feature matrix from dataframe"""
        tfidf_features = self.tfidf.transform(df['text'])
        bow_features = self.bow.transform(df['text'])
        return np.hstack([
            tfidf_features.toarray(),
            bow_features.toarray(),
            df[['lexicon_pos', 'lexicon_neg']].values
        ])
    
    def train(self, train_csv, val_csv):
        """Train the SVM classifier"""
        train_paths = [
            train_csv,
            f"data/processed/{train_csv}",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        f"data/processed/{train_csv}")
        ]
        
        val_paths = [
            val_csv,
            f"data/processed/{val_csv}",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        f"data/processed/{val_csv}")
        ]
        
        for path in train_paths:
            if os.path.exists(path):
                print(f"Loading training data from: {path}")
                train_df = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError(f"Could not find training data: {train_csv}")
        
        for path in val_paths:
            if os.path.exists(path):
                print(f"Loading validation data from: {path}")
                val_df = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError(f"Could not find validation data: {val_csv}")
        
        X_train = self.prepare_features(train_df)
        X_val = self.prepare_features(val_df)
        y_train = train_df['polarity'].map(self.label_map)
        y_val = val_df['polarity'].map(self.label_map)
        
        # Using CalibratedClassifierCV for probability estimates
        self.clf = CalibratedClassifierCV(
            LinearSVC(
                class_weight='balanced',
                max_iter=10000,
                random_state=42,
                dual=False
            ),
            method='sigmoid'
        )
        self.clf.fit(X_train, y_train)
        
        metrics = {
            'training': self._evaluate(X_train, y_train, 'training'),
            'validation': self._evaluate(X_val, y_val, 'validation')
        }
        
        # Create output directory if it doesn't exist
        os.makedirs("outputs/traditional", exist_ok=True)
        
        # Save model and metrics
        model_path = f"outputs/traditional/{self.domain}_svm_classifier.joblib"
        metrics_path = f"outputs/traditional/{self.domain}_svm_metrics.json"
        
        print(f"Saving model to: {model_path}")
        dump(self.clf, model_path)
        
        print(f"Saving metrics to: {metrics_path}")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _evaluate(self, X, y_true, split_name):
        """Evaluate model performance with well-formatted confusion matrix"""
        y_pred = self.clf.predict(X)
        cm = confusion_matrix(y_true, y_pred)
        
        # Format confusion matrix with labels
        formatted_cm = []
        labels = ["positive", "negative", "neutral"]
        
        # Add a header row with the predicted labels
        header = ["Actual/Predicted"] + labels
        formatted_cm.append(header)
        
        # Add rows with actual labels and values
        for i, label in enumerate(labels):
            row = [label] + [int(cm[i, j]) for j in range(len(labels))]
            formatted_cm.append(row)
            
        # Calculate per-class precision and recall
        per_class_metrics = {}
        for i, label in enumerate(labels):
            # Avoiding division by zero
            if np.sum(cm[i, :]) > 0:
                recall = cm[i, i] / np.sum(cm[i, :])
            else:
                recall = 0.0
                
            if np.sum(cm[:, i]) > 0:
                precision = cm[i, i] / np.sum(cm[:, i])
            else:
                precision = 0.0
                
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
                
            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return {
            f'{split_name}_accuracy': accuracy_score(y_true, y_pred),
            f'{split_name}_precision_macro': precision_score(y_true, y_pred, average='macro'),
            f'{split_name}_recall_macro': recall_score(y_true, y_pred, average='macro'),
            f'{split_name}_f1_macro': f1_score(y_true, y_pred, average='macro'),
            f'{split_name}_confusion_matrix': cm.tolist(),
            f'{split_name}_formatted_confusion_matrix': formatted_cm,
            f'{split_name}_per_class_metrics': per_class_metrics
        }
    
    def predict_from_test_data(self, test_csv):
        """Predict polarities for preprocessed test data"""
        # Try multiple possible locations for test CSV
        test_paths = [
            test_csv,
            f"data/processed/{test_csv}",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       f"data/processed/{test_csv}")
        ]
        
        # Find test CSV
        for path in test_paths:
            if os.path.exists(path):
                print(f"Loading test data from: {path}")
                test_df = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError(f"Could not find test data: {test_csv}")
        
        X_test = self.prepare_features(test_df)
        y_pred = self.clf.predict(X_test)
        test_df['predicted_polarity'] = [self.reverse_label_map[p] for p in y_pred]
        return test_df

def print_table_header(title, width=70):
    """Print a stylized table header"""
    print("\n" + "┌" + "─" * (width - 2) + "┐")
    print("│" + title.center(width - 2) + "│")
    print("└" + "─" * (width - 2) + "┘")

def print_metrics_table(title, metrics, labels=["positive", "negative", "neutral"]):
    """Print a stylized metrics table with headers and rows"""
    width = 70
    col_width = (width - 20) // 3
    
    # Print table header
    print_table_header(title, width)
    
    # Print column headers
    print("┌" + "─" * 12 + "┬" + "─" * col_width + "┬" + "─" * col_width + "┬" + "─" * col_width + "┐")
    print("│ " + "CLASS".ljust(10) + " │ " + "PRECISION".center(col_width - 2) + " │ " + 
          "RECALL".center(col_width - 2) + " │ " + "F1".center(col_width - 2) + " │")
    print("├" + "─" * 12 + "┼" + "─" * col_width + "┼" + "─" * col_width + "┼" + "─" * col_width + "┤")
    
    # Print rows
    for label in labels:
        print("│ " + label.ljust(10) + " │ " + 
              f"{metrics[label]['precision']:.4f}".center(col_width - 2) + " │ " + 
              f"{metrics[label]['recall']:.4f}".center(col_width - 2) + " │ " + 
              f"{metrics[label]['f1']:.4f}".center(col_width - 2) + " │")
    
    # Print table footer
    print("└" + "─" * 12 + "┴" + "─" * col_width + "┴" + "─" * col_width + "┴" + "─" * col_width + "┘")

def print_confusion_matrix(title, matrix, labels=["positive", "negative", "neutral"]):
    """Print a stylized confusion matrix with headers and rows"""
    width = 70
    col_width = (width - 20) // len(labels)
    
    # Print table header
    print_table_header(title, width)
    
    # Print column headers
    print("┌" + "─" * 18 + "┬" + "┬".join(["─" * col_width] * len(labels)) + "┐")
    header = "│ " + "ACTUAL/PREDICTED".ljust(16) + " │"
    for label in labels:
        header += " " + label.center(col_width - 2) + " │"
    print(header)
    print("├" + "─" * 18 + "┼" + "┼".join(["─" * col_width] * len(labels)) + "┤")
    
    # Print rows (skip header row, which is at index 0)
    for i, label in enumerate(labels):
        row = "│ " + label.ljust(16) + " │"
        for j in range(len(labels)):
            cell_value = matrix[i+1][j+1] # +1 to skip headers
            row += " " + str(cell_value).center(col_width - 2) + " │"
        print(row)
    
    # Print table footer
    print("└" + "─" * 18 + "┴" + "┴".join(["─" * col_width] * len(labels)) + "┘")

if __name__ == "__main__":
    # Add the project root to the Python path
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Make sure output directories exist
    os.makedirs("outputs/traditional", exist_ok=True)
    
    domains = ['laptop', 'restaurant']
    
    # Train SVM models for each domain
    for domain in domains:
        print("\n" + "=" * 70)
        print(f"TRAINING {domain.upper()} SVM MODEL")
        print("=" * 70)
        
        try:
            svm_classifier = SVM_Classifier(domain)
            metrics = svm_classifier.train(
                f"{domain}_train_features.csv",
                f"{domain}_val_features.csv"
            )
            
            # Print a nicely formatted summary of the metrics
            print_table_header(f"ASPECT-BASED SENTIMENT ANALYSIS - {domain.upper()} DOMAIN")
            
            # Print overall metrics
            print_table_header("OVERALL PERFORMANCE METRICS")
            print(f"Training Accuracy:   {metrics['training']['training_accuracy']:.4f}")
            print(f"Validation Accuracy: {metrics['validation']['validation_accuracy']:.4f}")
            print(f"Training F1 (macro): {metrics['training']['training_f1_macro']:.4f}")
            print(f"Validation F1 (macro): {metrics['validation']['validation_f1_macro']:.4f}")
            
            # Print per-class metrics table
            print_metrics_table("PER-CLASS METRICS (VALIDATION)", 
                              metrics['validation']['validation_per_class_metrics'])
            
            # Print confusion matrices
            print_confusion_matrix("TRAINING CONFUSION MATRIX",
                                 metrics['training']['training_formatted_confusion_matrix'])
            
            print_confusion_matrix("VALIDATION CONFUSION MATRIX",
                                 metrics['validation']['validation_formatted_confusion_matrix'])
                
        except Exception as e:
            print(f"Error training {domain} SVM model: {e}")
    
    # Generate predictions for each domain
    for domain in domains:
        try:
            print("\n" + "=" * 70)
            print(f"GENERATING PREDICTIONS FOR {domain.upper()}")
            print("=" * 70)
            
            model_paths = [
                f"{domain}_svm_classifier.joblib",
                f"outputs/traditional/{domain}_svm_classifier.joblib",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           f"outputs/traditional/{domain}_svm_classifier.joblib")
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    print(f"Loading model from: {path}")
                    svm_classifier = SVM_Classifier(domain)
                    svm_classifier.clf = load(path)
                    break
            else:
                raise FileNotFoundError(f"Could not find model file for {domain}")
            
            test_file = f"{domain}_test_features_unlabeled.csv"
            predictions = svm_classifier.predict_from_test_data(test_file)
            
            sentiment_counts = predictions['predicted_polarity'].value_counts()
            
            # Print prediction distribution table
            print_table_header("PREDICTION DISTRIBUTION")
            print("┌" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┐")
            print("│ " + "SENTIMENT".ljust(10) + " │ " + "COUNT".center(10) + " │ " + "PERCENTAGE".center(10) + " │")
            print("├" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")
            
            for sentiment, count in sentiment_counts.items():
                percentage = count/len(predictions)*100
                print(f"│ {sentiment.ljust(10)} │ {str(count).center(10)} │ {f'{percentage:.1f}%'.center(10)} │")
            
            print("└" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┘")
            
            # Save predictions
            output_path = f"outputs/traditional/{domain}_svm_test_predictions.csv"
            print(f"Saving predictions to: {output_path}")
            predictions[['sentence', 'aspect', 'predicted_polarity']].to_csv(
                output_path,
                index=False
            )
            
            # Show sample predictions table
            print_table_header("SAMPLE PREDICTIONS")
            print("┌" + "─" * 40 + "┬" + "─" * 15 + "┬" + "─" * 12 + "┐")
            print("│ " + "SENTENCE".ljust(38) + " │ " + "ASPECT".center(13) + " │ " + "SENTIMENT".center(10) + " │")
            print("├" + "─" * 40 + "┼" + "─" * 15 + "┼" + "─" * 12 + "┤")
            
            sample = predictions[['sentence', 'aspect', 'predicted_polarity']].head(5)
            for _, row in sample.iterrows():
                sentence = row['sentence']
                if len(sentence) > 37:
                    sentence = sentence[:34] + "..."
                
                print(f"│ {sentence.ljust(38)} │ {row['aspect'].center(13)} │ {row['predicted_polarity'].center(10)} │")
            
            print("└" + "─" * 40 + "┴" + "─" * 15 + "┴" + "─" * 12 + "┘")
            
            print(f"{domain.upper()} SVM predictions saved successfully!")
            
        except Exception as e:
            print(f"Error generating predictions for {domain}: {e}")