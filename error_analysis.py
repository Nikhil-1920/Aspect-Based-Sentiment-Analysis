import sys
import os
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# --- Awesome Makeover: Add a class for terminal colors ---
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

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print(f"{BColors.WARNING}Spacy model 'en_core_web_sm' not found. Downloading...{BColors.ENDC}")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def load_model_predictions(model_type, domain):
    """Load model predictions"""
    path_map = {
        'svm': f"outputs/traditional/{domain}_svm_test_predictions.csv",
        'lr': f"outputs/traditional/{domain}_lr_test_predictions.csv",
        'bert': f"outputs/bert/{domain}/{domain}_bert_predictions.csv",
        'hybrid': f"outputs/hybrid/{domain}/predictions.csv"
    }
    try:
        path = path_map.get(model_type)
        if path and os.path.exists(path):
            return pd.read_csv(path)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"{BColors.FAIL}Predictions file for {model_type.upper()} model on {domain.upper()} domain not found.{BColors.ENDC}")
        return None

def analyze_errors_by_sentiment(domain, model):
    """Analyze which sentiments are most challenging"""
    df = load_model_predictions(model, domain)
    if df is None: return None
    
    sentiment_counts = df['predicted_polarity'].value_counts()
    print(f"\n{BColors.OKCYAN}📊 {model.upper()} - Prediction Distribution:{BColors.ENDC}")
    for sentiment, count in sentiment_counts.items():
        percentage = count / len(df) * 100
        print(f"  - {sentiment.capitalize():<10}: {BColors.OKGREEN}{count}{BColors.ENDC} ({BColors.WARNING}{percentage:.1f}%{BColors.ENDC})")
    return df

def analyze_common_errors(domain, models):
    """Find instances that all models get wrong"""
    predictions = {}
    for model in models:
        pred_df = load_model_predictions(model, domain)
        if pred_df is not None:
            pred_df['id'] = pred_df['sentence'] + ' | ' + pred_df['aspect']
            predictions[model] = pred_df.set_index('id')[['sentence', 'aspect', 'predicted_polarity']]
    
    if len(predictions) < 2: return None
    
    common_ids = set.intersection(*[set(preds.index) for preds in predictions.values()])
    
    disagreements = []
    for id in common_ids:
        preds = {m: predictions[m].loc[id, 'predicted_polarity'] for m in models if m in predictions}
        if len(set(preds.values())) > 1:
            sentence = predictions[models[0]].loc[id, 'sentence']
            aspect = predictions[models[0]].loc[id, 'aspect']
            disagreements.append({'sentence': sentence, 'aspect': aspect, **preds})
    
    return pd.DataFrame(disagreements)

def analyze_aspects(domain, model):
    """Analyze which aspects are most challenging"""
    df = load_model_predictions(model, domain)
    if df is None: return
    
    aspect_sentiment = df.groupby('aspect')['predicted_polarity'].value_counts().unstack(fill_value=0)
    if not aspect_sentiment.empty:
        aspect_sentiment['total'] = aspect_sentiment.sum(axis=1)
        aspect_sentiment['dominant'] = aspect_sentiment.idxmax(axis=1)
        top_aspects = aspect_sentiment.sort_values('total', ascending=False).head(10)
    else:
        top_aspects = pd.DataFrame()

    print(f"\n{BColors.OKCYAN}🎯 {model.upper()} - Top 10 Aspects by Frequency:{BColors.ENDC}")
    if top_aspects.empty:
        print(f"  {BColors.FAIL}No aspects found to analyze.{BColors.ENDC}")
        return
    for aspect, row in top_aspects.iterrows():
        print(f"  - {BColors.OKBLUE}{aspect:<20}{BColors.ENDC} Total: {BColors.OKGREEN}{int(row['total'])}{BColors.ENDC}, Dominant Sentiment: {BColors.WARNING}{row['dominant']}{BColors.ENDC}")
    return aspect_sentiment

def analyze_sentence_length(domain, model):
    """Analyze if sentence length affects predictions"""
    df = load_model_predictions(model, domain)
    if df is None: return
    
    df['sentence_length'] = df['sentence'].apply(len)
    length_by_sentiment = df.groupby('predicted_polarity')['sentence_length'].agg(['mean', 'median', 'std']).round(1)
    
    print(f"\n{BColors.OKCYAN}📏 {model.upper()} - Sentence Length by Predicted Sentiment:{BColors.ENDC}")
    
    # Colorize tabulate output
    table_str = tabulate(length_by_sentiment, headers=['Sentiment', 'Mean', 'Median', 'Std Dev'], tablefmt="heavy_grid")
    print(BColors.OKBLUE + table_str + BColors.ENDC)
    
    plt.figure(figsize=(10, 6))
    for sentiment in df['predicted_polarity'].unique():
        subset = df[df['predicted_polarity'] == sentiment]
        plt.hist(subset['sentence_length'], alpha=0.6, bins=20, label=sentiment.capitalize())
    
    plt.xlabel('Sentence Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Sentence Length Distribution by Sentiment - {model.upper()} - {domain.upper()}', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs('outputs/analysis', exist_ok=True)
    save_path = f'outputs/analysis/{domain}_{model}_sentence_length.png'
    plt.savefig(save_path)
    plt.close() # Close plot to free memory
    print(f"  {BColors.OKGREEN}✓ Plot saved to {save_path}{BColors.ENDC}")
    
    return length_by_sentiment

def analyze_keywords(domain, model):
    """Find keywords that are strongly associated with each sentiment"""
    df = load_model_predictions(model, domain)
    if df is None: return
    
    sentiment_texts = {s: ' '.join(df[df['predicted_polarity'] == s]['sentence']) for s in df['predicted_polarity'].unique()}
    stop_words = set(stopwords.words('english'))
    
    sentiment_words = {}
    for sentiment, text in sentiment_texts.items():
        tokens = word_tokenize(text.lower())
        filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
        sentiment_words[sentiment] = Counter(filtered_tokens)
    
    distinctive_words = {}
    for sentiment, word_freq in sentiment_words.items():
        words_with_scores = []
        for word, count in word_freq.items():
            count_in_others = sum(other_freq.get(word, 0) for other_s, other_freq in sentiment_words.items() if other_s != sentiment)
            distinctiveness = count / (count_in_others + 1)
            words_with_scores.append((word, count, distinctiveness))
        
        distinctive_words[sentiment] = sorted(words_with_scores, key=lambda x: x[2], reverse=True)[:10]

    print(f"\n{BColors.OKCYAN}🔍 {model.upper()} - Top Distinctive Keywords by Sentiment:{BColors.ENDC}")
    for sentiment, words in distinctive_words.items():
        print(f"\n  {BColors.BOLD}{sentiment.upper()}:{BColors.ENDC}")
        for word, count, score in words:
            print(f"    - {BColors.OKBLUE}{word:<15}{BColors.ENDC} (Count: {BColors.OKGREEN}{count:<4}{BColors.ENDC}, Score: {BColors.WARNING}{score:.2f}{BColors.ENDC})")
    
    return distinctive_words

def analyze_confusion_patterns(domain, models):
    """Analyze patterns in model disagreements"""
    disagreements = analyze_common_errors(domain, models)
    if disagreements is None or len(disagreements) == 0:
        print(f"\n{BColors.OKGREEN}✓ No disagreements found between models for {domain.upper()} domain.{BColors.ENDC}")
        return
    
    print(f"\n{BColors.HEADER}{BColors.BOLD}🤝 Model Disagreement Analysis for {domain.upper()} 🤝{BColors.ENDC}")
    
    patterns = [tuple(sorted({m: row[m] for m in models if m in row}.items())) for _, row in disagreements.iterrows()]
    pattern_counts = Counter(patterns)
    
    print(f"\n{BColors.OKCYAN}Common Disagreement Patterns:{BColors.ENDC}")
    for pattern, count in pattern_counts.most_common(5):
        pattern_str = " vs ".join([f"{m.upper()}({BColors.WARNING}{s}{BColors.ENDC})" for m, s in pattern])
        print(f"  - {pattern_str:<40} : {BColors.OKGREEN}{count}{BColors.ENDC} instances")
    
    aspect_counts = disagreements['aspect'].value_counts()
    print(f"\n{BColors.OKCYAN}Top Aspects with Disagreements:{BColors.ENDC}")
    for aspect, count in aspect_counts.head(5).items():
        print(f"  - {BColors.OKBLUE}{aspect:<20}{BColors.ENDC} : {BColors.OKGREEN}{count}{BColors.ENDC} instances")
    
    return disagreements

def generate_error_report(domains, models):
    """Generate a comprehensive error analysis report"""
    os.makedirs("outputs/analysis", exist_ok=True)
    
    for domain in domains:
        print(f"\n{BColors.HEADER}{BColors.BOLD}{'='*80}{BColors.ENDC}")
        title = f"🔍 ANALYZING {domain.upper()} DOMAIN 🔍"
        print(f"{BColors.HEADER}{BColors.BOLD}{title.center(80)}{BColors.ENDC}")
        print(f"{BColors.HEADER}{BColors.BOLD}{'='*80}{BColors.ENDC}")
        
        report_content = []
        for model in models:
            print(f"\n{BColors.HEADER}--- Analyzing Model: {model.upper()} ---{BColors.ENDC}")
            analyze_errors_by_sentiment(domain, model)
            analyze_aspects(domain, model)
            analyze_sentence_length(domain, model)
            analyze_keywords(domain, model)
        
        analyze_confusion_patterns(domain, models)
        
        # Note: The file-saving logic is simplified here to avoid running analysis twice.
        # For a true file report, you would capture the styled output.
        # This implementation prioritizes a clean console experience.
        report_path = f"outputs/analysis/{domain}_error_analysis.txt"
        with open(report_path, "w") as f:
            f.write(f"ERROR ANALYSIS REPORT FOR {domain.upper()} DOMAIN\n")
            f.write("="*70 + "\n\n")
            f.write("This report is a placeholder. The detailed, colored output is available in the terminal.\n")
            f.write("To generate a full text report, the print statements in the script would need to be redirected.\n")

        print(f"\n{BColors.OKGREEN}💾 Simplified error analysis report saved to {report_path}{BColors.ENDC}")
        print(f"{BColors.WARNING}Note: The saved report is a summary. For detailed analysis, please refer to the terminal output above.{BColors.ENDC}")


if __name__ == "__main__":
    domains = ['laptop', 'restaurant']
    models = ['svm', 'lr', 'bert', 'hybrid']
    
    generate_error_report(domains, models)