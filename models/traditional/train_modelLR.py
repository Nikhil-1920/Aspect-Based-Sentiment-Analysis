import sys
import os
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from joblib import dump, load

# --- FIX 1: Correctly add the project root to the Python path ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# --- FIX 2: Simplify the AspectExtractor import ---
from models.utils.aspect_extractor import AspectExtractor

# --- AWESOME MAKEOVER 1: Add a class for terminal colors ---
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- AWESOME MAKEOVER 2: Create a dedicated, stylish print function ---
def print_awesome_metrics(domain_name, metrics, reverse_label_map):
    domain_header = f" {domain_name.upper()} VALIDATION METRICS "
    print(f"\n{BColors.HEADER}{BColors.BOLD}{'='*80}{BColors.ENDC}")
    print(f"{BColors.HEADER}{BColors.BOLD}{domain_header:^80}{BColors.ENDC}")
    print(f"{BColors.HEADER}{BColors.BOLD}{'='*80}{BColors.ENDC}\n")

    # --- Overall Metrics ---
    print(f"{BColors.OKCYAN}{BColors.BOLD}📈 Overall Performance:{BColors.ENDC}")
    print(f"  {'Accuracy:':<20} {BColors.OKGREEN}{metrics['validation_accuracy']:.4f}{BColors.ENDC}")
    print(f"  {'F1-Score (Macro):':<20} {BColors.OKGREEN}{metrics['validation_f1_macro']:.4f}{BColors.ENDC}")
    print(f"  {'Precision (Macro):':<20} {metrics['validation_precision_macro']:.4f}")
    print(f"  {'Recall (Macro):':<20} {metrics['validation_recall_macro']:.4f}\n")

    # --- Confusion Matrix ---
    print(f"{BColors.OKCYAN}{BColors.BOLD}📊 Confusion Matrix:{BColors.ENDC}")
    cm = metrics['validation_confusion_matrix']
    labels = list(reverse_label_map.values())
    
    label = "Actual \\ Predicted"
    header = f" {label:<18} |"  
    # header = f" {'Actual \\ Predicted':<18} |"
    for label in labels:
        header += f" {label.capitalize():^10} |"
    print(f"{BColors.WARNING}{header}{BColors.ENDC}")
    print(f"{BColors.WARNING}{'-' * len(header)}{BColors.ENDC}")

    for i, row in enumerate(cm):
        row_str = f" {labels[i].capitalize():<18} |"
        for val in row:
            row_str += f" {val:^10} |"
        print(row_str)
    print("")

    # --- Per-Class Metrics ---
    print(f"{BColors.OKCYAN}{BColors.BOLD}🎯 Per-Class Performance:{BColors.ENDC}")
    per_class_metrics = metrics['validation_per_class_metrics']
    
    class_header = f" {'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
    print(f"{BColors.WARNING}{class_header}{BColors.ENDC}")
    print(f"{BColors.WARNING}{'-' * len(class_header)}{BColors.ENDC}")
    
    for label, scores in per_class_metrics.items():
        p, r, f1 = scores['precision'], scores['recall'], scores['f1']
        print(f" {label.capitalize():<12} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
    
    print(f"\n{BColors.HEADER}{BColors.BOLD}{'='*80}{BColors.ENDC}\n")


class ABSA_Classifier:
    def __init__(self, domain):
        self.domain = domain
        self.aspect_extractor = AspectExtractor()
        
        tfidf_paths = [
            f"{domain}_tfidf_vectorizer.joblib",
            f"data/processed/{domain}_tfidf_vectorizer.joblib", 
            os.path.join(project_root, f"data/processed/{domain}_tfidf_vectorizer.joblib")
        ]
        
        bow_paths = [
            f"{domain}_bow_vectorizer.joblib",
            f"data/processed/{domain}_bow_vectorizer.joblib",
            os.path.join(project_root, f"data/processed/{domain}_bow_vectorizer.joblib")
        ]
        
        for path in tfidf_paths:
            try:
                self.tfidf = load(path)
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(f"Could not find TF-IDF vectorizer for {domain} in any location")
        
        for path in bow_paths:
            try:
                self.bow = load(path)
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
        """Train the classifier"""
        train_path = os.path.join(project_root, "data/processed", train_csv)
        val_path = os.path.join(project_root, "data/processed", val_csv)

        if not os.path.exists(train_path):
             raise FileNotFoundError(f"Could not find training data: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Could not find validation data: {val_path}")

        print(f"Loading training data from: {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"Loading validation data from: {val_path}")
        val_df = pd.read_csv(val_path)

        X_train = self.prepare_features(train_df)
        X_val = self.prepare_features(val_df)
        y_train = train_df['polarity'].map(self.label_map)
        y_val = val_df['polarity'].map(self.label_map)
        
        self.clf = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            C=0.1
        )
        self.clf.fit(X_train, y_train)
        
        metrics = {
            'training': self._evaluate(X_train, y_train, 'training'),
            'validation': self._evaluate(X_val, y_val, 'validation')
        }
        
        output_dir = os.path.join(project_root, "outputs", "traditional")
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f"{self.domain}_lr_classifier.joblib")
        dump(self.clf, model_path)
        print(f"{BColors.OKGREEN}✅ Model saved to {model_path}{BColors.ENDC}")
        
        metrics_path = os.path.join(output_dir, f"{self.domain}_lr_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"{BColors.OKGREEN}✅ Metrics saved to {metrics_path}{BColors.ENDC}")
        
        return metrics
    
    def _evaluate(self, X, y_true, split_name):
        """Evaluate model performance with per-class metrics"""
        y_pred = self.clf.predict(X)
        
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_metrics = {}
        for i, label in enumerate(self.reverse_label_map.values()):
            per_class_metrics[label] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
        
        return {
            f'{split_name}_accuracy': accuracy_score(y_true, y_pred),
            f'{split_name}_precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            f'{split_name}_recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            f'{split_name}_f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            f'{split_name}_confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            f'{split_name}_per_class_metrics': per_class_metrics
        }
    
    def predict_from_text(self, sentences, domain):
        """Predict aspects and polarities for raw sentences"""
        if not self.clf:
            model_path = os.path.join(project_root, "outputs", "traditional", f"{self.domain}_lr_classifier.joblib")
            try:
                self.clf = load(model_path)
            except FileNotFoundError:
                raise ValueError(f"Model not trained and not found at {model_path}. Call train() first.")

        results = []
        for sentence in sentences:
            aspects = self.aspect_extractor.extract_aspects(sentence, domain)
            
            if not aspects:
                aspects = ['general']
            
            for aspect in aspects:
                text = f"{sentence} {aspect}"
                
                tfidf_feats = self.tfidf.transform([text])
                bow_feats = self.bow.transform([text])
                lexicon_feats = {'lexicon_pos': 0, 'lexicon_neg': 0}
                
                X = np.hstack([
                    tfidf_feats.toarray(),
                    bow_feats.toarray(),
                    np.array([[lexicon_feats['lexicon_pos'], lexicon_feats['lexicon_neg']]])
                ])
                
                polarity_idx = self.clf.predict(X)[0]
                polarity = self.reverse_label_map[polarity_idx]
                
                results.append({
                    'sentence': sentence,
                    'aspect': aspect,
                    'predicted_polarity': polarity
                })
        
        return pd.DataFrame(results)

    def predict_from_test_data(self, test_csv):
        """Predict polarities for preprocessed test data with extracted aspects"""
        test_path = os.path.join(project_root, "data/processed", test_csv)
        if not os.path.exists(test_path):
             raise FileNotFoundError(f"Could not find test data: {test_path}")

        print(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path)
            
        X_test = self.prepare_features(test_df)
        y_pred = self.clf.predict(X_test)
        test_df['predicted_polarity'] = [self.reverse_label_map[p] for p in y_pred]
        return test_df

if __name__ == "__main__":
    for domain in ['laptop', 'restaurant']:
        print(f"\n{BColors.OKBLUE}⚙️  Training {domain.upper()} model...{BColors.ENDC}")
        try:
            absa = ABSA_Classifier(domain)
            metrics = absa.train(f"{domain}_train_features.csv", f"{domain}_val_features.csv")
            # --- AWESOME MAKEOVER 3: Use the new print function ---
            print_awesome_metrics(domain, metrics['validation'], absa.reverse_label_map)
        except Exception as e:
            print(f"{BColors.FAIL}Error training {domain} model: {e}{BColors.ENDC}")
            
    print("\n" + "="*80 + "\n")

    for domain in ['laptop', 'restaurant']:
        print(f"\n{BColors.OKBLUE}💡 Predicting for {domain.upper()} test data...{BColors.ENDC}")
        try:
            test_file = f"{domain}_test_features_unlabeled.csv"
            
            absa = ABSA_Classifier(domain)
            model_path = os.path.join(project_root, "outputs", "traditional", f"{domain}_lr_classifier.joblib")
            
            print(f"Loading model from: {model_path}")
            absa.clf = load(model_path)
            
            test_df = absa.predict_from_test_data(test_file)
            
            output_path = os.path.join(project_root, "outputs", "traditional", f"{domain}_lr_test_predictions.csv")
            test_df[['sentence', 'aspect', 'predicted_polarity']].to_csv(output_path, index=False)
            print(f"{BColors.OKGREEN}✅ Predictions saved to {output_path}{BColors.ENDC}")
        except Exception as e:
            print(f"{BColors.FAIL}Error predicting for {domain}: {e}{BColors.ENDC}")