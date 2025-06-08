import sys
import os
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize
import spacy
import nltk
import json
import pickle
from joblib import dump

try:
    swn.senti_synsets('good')
except:
    print("Downloading NLTK components...")
    nltk.download('sentiwordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

# Initialize spaCy for advanced NLP features
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    sys.exit()


# --- UNUSED FUNCTIONS (kept as requested for potential external use) ---

def preprocess_domain(xml_file, domain):
    """Full preprocessing pipeline per domain"""
    # Parse and filter
    df = parse_xml(xml_file, domain)
    df = df[df['polarity'].isin(['positive', 'negative', 'neutral'])]
    
    # Extract and save aspects
    aspects = set(df['aspect'].str.lower())
    os.makedirs('domain_data', exist_ok=True)
    with open(f'domain_data/{domain}_aspects.json', 'w') as f:
        json.dump(list(aspects), f)
    
    # Split data (grouped by sentence)
    sentences = df.groupby('sentence').agg(list)
    train_sents, val_sents = train_test_split(
        sentences,
        test_size=0.2,
        stratify=sentences['polarity'].apply(tuple),
        random_state=42
    )
    
    # Flatten and save splits
    for split, name in [(train_sents, 'train'), (val_sents, 'val')]:
        split_df = split.explode(['aspect', 'polarity'])
        split_df.to_csv(f'domain_data/{domain}_{name}_features.csv', index=False)
    
    # Train vectorizers only on train data
    train_df = pd.read_csv(f'domain_data/{domain}_train_features.csv')
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=15000)
    bow = CountVectorizer(ngram_range=(1, 3), max_features=15000)
    
    # Fit and save vectorizers
    tfidf.fit(train_df['sentence'] + " " + train_df['aspect'])
    bow.fit(train_df['sentence'] + " " + train_df['aspect'])
    dump(tfidf, f'domain_data/{domain}_tfidf.joblib')
    dump(bow, f'domain_data/{domain}_bow.joblib')

def process_test_data(test_xml, domain, tfidf, bow):
    """Preprocess test data using trained TF-IDF and BoW from train data."""
    test_df = parse_xml(test_xml, domain)
    
    test_df = test_df[test_df['polarity'].isin(['positive', 'negative', 'neutral'])]
    
    test_df = create_features(test_df)
    
    # Transform using trained vectorizers
    X_test_tfidf = tfidf.transform(test_df['text'])
    X_test_bow = bow.transform(test_df['text'])

    X_test = np.hstack([
        X_test_tfidf.toarray(),
        X_test_bow.toarray(),
        test_df[['lexicon_pos', 'lexicon_neg']].values
    ])
    
    y_test = test_df['polarity'].map({'positive': 0, 'negative': 1, 'neutral': 2})
    
    test_df.to_csv("test_features.csv", index=False)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

# --- SCRIPT LOGIC (functions used in the main execution block) ---

def parse_xml(xml_file, domain):
    """Parse XML files and extract aspect-sentence pairs (aspect terms only)"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    
    for sentence in root.findall('sentence'):
        text_element = sentence.find('text')
        if text_element is None or text_element.text is None:
            continue
        text = text_element.text.replace('&quot;', '"').replace('&apos;', "'").strip()
        aspects = []

        for term in sentence.findall('aspectTerms/aspectTerm'):
            aspect_term = term.get('term')
            polarity = term.get('polarity')
            if aspect_term and polarity:
                aspects.append((aspect_term.replace('&quot;', '"').strip(), polarity))

        for aspect, polarity in aspects:
            data.append({
                'sentence': text,
                'aspect': aspect,
                'polarity': polarity.lower()
            })
    
    return pd.DataFrame(data)

def get_lexicon_features(text):
    """Extract sentiment lexicon scores using SentiWordNet"""
    pos_tags = pos_tag(word_tokenize(text))
    pos_scores = []
    for word, tag in pos_tags:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            pos_scores.append(synsets[0].pos_score())
    return {
        'lexicon_pos': np.mean(pos_scores) if pos_scores else 0,
        'lexicon_neg': np.mean([1 - score for score in pos_scores]) if pos_scores else 0
    }

def get_dependency_features(sentence, aspect):
    """Extract syntactic features using spaCy"""
    doc = nlp(sentence)
    aspect_tokens = set(aspect.lower().split())
    dep_features = []
    
    for token in doc:
        if token.text.lower() in aspect_tokens:
            dep_features.extend([
                token.dep_,
                token.head.text,
                token.head.pos_
            ])
    return {'dependency': ' '.join(dep_features) if dep_features else ''}

def create_features(df):
    """Create multiple feature sets"""
    # Basic text features
    df['text'] = df['sentence'] + " " + df['aspect']
    
    # Sentiment lexicon features
    lexicon_features = df['sentence'].apply(get_lexicon_features).apply(pd.Series)
    
    # Dependency features
    dep_features = df.apply(
        lambda x: get_dependency_features(x['sentence'], x['aspect']), axis=1
    ).apply(pd.Series)
    
    return df.join(lexicon_features).join(dep_features)

def vectorize_data(train_df, val_df):
    """Create multiple feature representations"""
    # TF-IDF with n-grams
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=15000,
        stop_words='english'
    )
    X_train_tfidf = tfidf.fit_transform(train_df['text'])
    X_val_tfidf = tfidf.transform(val_df['text'])
    
    # Bag-of-Words
    bow = CountVectorizer(
        ngram_range=(1, 3),
        max_features=15000
    )
    X_train_bow = bow.fit_transform(train_df['text'])
    X_val_bow = bow.transform(val_df['text'])
    
    # Combine all features
    X_train = np.hstack([
        X_train_tfidf.toarray(),
        X_train_bow.toarray(),
        train_df[['lexicon_pos', 'lexicon_neg']].values
    ])
    
    X_val = np.hstack([
        X_val_tfidf.toarray(),
        X_val_bow.toarray(),
        val_df[['lexicon_pos', 'lexicon_neg']].values
    ])
    
    return X_train, X_val, tfidf, bow

def extract_aspect_terms(xml_file, domain):
    """Extract unique aspect terms from training XML files"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    aspects = set()
    
    for sentence in root.findall('sentence'):
        for term in sentence.findall('aspectTerms/aspectTerm'):
            aspect = term.get('term').lower().strip()
            aspects.add(aspect)
    
    return {domain: list(aspects)}    

if __name__ == "__main__":
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ All output files will be saved to: '{output_dir}/'")

    domains = [("./data/raw/Laptop_Train_v2.xml", "laptop"), ("./data/raw/Restaurants_Train_v2.xml", "restaurant")]
    all_aspects = {}

    for xml_file, domain in domains:
        title = f"Processing domain: {domain.upper()}"
        
        # Print the title with a matching line of '=' characters above and below
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}")
        
        df = parse_xml(xml_file, domain)
        df = df[df['polarity'].isin(['positive', 'negative', 'neutral'])]
        
        sentences = df.groupby('sentence').agg(list).reset_index()
        class_counts = sentences['polarity'].apply(lambda p: tuple(sorted(p))).value_counts()
        valid_classes = class_counts[class_counts > 1].index
        sentences = sentences[sentences['polarity'].apply(lambda p: tuple(sorted(p))).isin(valid_classes)]
        
        train_sents, val_sents = train_test_split(
            sentences, test_size=0.2, stratify=sentences['polarity'].apply(lambda p: tuple(sorted(p))), random_state=42
        )
        
        train_df = train_sents.explode(['aspect', 'polarity']).reset_index(drop=True)
        val_df = val_sents.explode(['aspect', 'polarity']).reset_index(drop=True)
        
        print("🔧 Creating features...")
        train_df = create_features(train_df)
        val_df = create_features(val_df)
        
        print("🧠 Training vectorizers and transforming data...")
        X_train, X_val, tfidf, bow = vectorize_data(train_df, val_df)
        
        # --- Save all files to the 'data/processed' directory ---
        print(f"💾 Saving processed files for '{domain}' domain...")
        dump(tfidf, os.path.join(output_dir, f"{domain}_tfidf_vectorizer.joblib"))
        dump(bow, os.path.join(output_dir, f"{domain}_bow_vectorizer.joblib"))
        
        train_df.to_csv(os.path.join(output_dir, f"{domain}_train_features.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, f"{domain}_val_features.csv"), index=False)
        
        np.save(os.path.join(output_dir, f"{domain}_X_train.npy"), X_train)
        np.save(os.path.join(output_dir, f"{domain}_X_val.npy"), X_val)
        np.save(os.path.join(output_dir, f"{domain}_y_train.npy"), train_df['polarity'].map({'positive':0, 'negative':1, 'neutral':2}).values)
        np.save(os.path.join(output_dir, f"{domain}_y_val.npy"), val_df['polarity'].map({'positive':0, 'negative':1, 'neutral':2}).values)
        
        domain_aspects = extract_aspect_terms(xml_file, domain)
        all_aspects.update(domain_aspects)
        print("...Done.")

    print("\n💾 Saving combined aspect candidates file...")
    with open(os.path.join(output_dir, "aspect_candidates.json"), "w") as f:
        json.dump(all_aspects, f, indent=4)

    print("\n✨ Preprocessing complete! ✨")

