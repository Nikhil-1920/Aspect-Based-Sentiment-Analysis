import sys
import os
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import sys
import json
import spacy
import nltk
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Add project root to Python path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Setup: Download NLTK and load spaCy ---
try:
    nltk.data.find('corpora/sentiwordnet.zip')
except nltk.downloader.DownloadError:
    print("Downloading NLTK resources (sentiwordnet, punkt, etc.)...")
    nltk.download('sentiwordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("❌ Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()

# --- Core Functions ---
def parse_xml(xml_file):
    """Parse XML files and extract sentence, aspect, and polarity."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    
    for sentence in root.findall('sentence'):
        text_element = sentence.find('text')
        if text_element is None or text_element.text is None:
            continue
        text = text_element.text.replace('&quot;', '"').replace('&apos;', "'").strip()
        
        for term in sentence.findall('aspectTerms/aspectTerm'):
            aspect = term.get('term')
            polarity = term.get('polarity')
            if aspect and polarity:
                data.append({
                    'sentence': text,
                    'aspect': aspect.replace('&quot;', '"').strip(),
                    'polarity': polarity.lower()
                })
    
    return pd.DataFrame(data)

def extract_aspect_terms(df, domain, output_dir):
    """Extract and save unique aspect terms for a domain to the specified directory."""
    aspects = set(df['aspect'].str.lower())
    aspects_path = os.path.join(output_dir, f'{domain}_aspects.json')
    with open(aspects_path, 'w') as f:
        json.dump(list(aspects), f, indent=4)
    print(f"   - Extracted {len(aspects)} unique aspects for '{domain}' domain.")
    return aspects

def preprocess_for_bert(xml_file, domain, output_dir):
    """Preprocess training data for a BERT model."""
    df = parse_xml(xml_file)
    df = df[df['polarity'].isin(['positive', 'negative', 'neutral'])]
    
    # Extract and save aspect terms to the BERT output directory
    extract_aspect_terms(df, domain, output_dir)
    
    # Group by sentence to ensure coherent train/validation split
    sentences = df.groupby('sentence').agg(list).reset_index()
    
    # Stratify split based on the combination of polarities in each sentence
    sentences['polarity_tuple'] = sentences['polarity'].apply(lambda p: tuple(sorted(p)))
    class_counts = sentences['polarity_tuple'].value_counts()
    valid_classes = class_counts[class_counts > 1].index
    sentences = sentences[sentences['polarity_tuple'].isin(valid_classes)]
    
    train_sents, val_sents = train_test_split(
        sentences, 
        test_size=0.2, 
        stratify=sentences['polarity_tuple'],
        random_state=42
    )
    
    train_df = train_sents.explode(['aspect', 'polarity']).reset_index(drop=True)
    val_df = val_sents.explode(['aspect', 'polarity']).reset_index(drop=True)
    
    train_df.to_csv(os.path.join(output_dir, f"{domain}_train_features.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"{domain}_val_features.csv"), index=False)
    
    print(f"   - Saved {len(train_df)} training and {len(val_df)} validation examples.")

def prepare_test_data(xml_file, domain, output_dir):
    """Prepare unlabeled test data by extracting aspects."""
    from models.utils.aspect_extractor import AspectExtractor
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    aspect_extractor = AspectExtractor()
    
    for sentence in root.findall('sentence'):
        text_element = sentence.find('text')
        if text_element is None or text_element.text is None:
            continue
        text = text_element.text.replace('&quot;', '"').replace('&apos;', "'").strip()
        
        aspects = aspect_extractor.extract_aspects(text, domain)
        
        for aspect in aspects:
            data.append({'sentence': text, 'aspect': aspect})
    
    test_df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, f"{domain}_test_features.csv")
    test_df.to_csv(output_path, index=False)
    print(f"   - Saved {len(test_df)} unlabeled test examples to '{output_path}'")

if __name__ == "__main__":
    # --- Define Paths ---
    raw_data_dir = 'data/raw'
    output_dir = 'data/processed/bert_data'
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ All BERT data will be saved to: '{output_dir}/'")

    data_files = [
        ("Laptop_Train_v2.xml", "laptop", "train"),
        ("Restaurants_Train_v2.xml", "restaurant", "train"),
        ("Laptops_Test_Data_PhaseA.xml", "laptop", "test"),
        ("Restaurants_Test_Data_PhaseA.xml", "restaurant", "test")
    ]
    
    for filename, domain, split_type in data_files:
        title = f"Processing {domain.upper()} {split_type.upper()} data"
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}")
        
        xml_file_path = os.path.join(raw_data_dir, filename)
        
        if not os.path.exists(xml_file_path):
            print(f"❌ Error: Data file not found at '{xml_file_path}'. Skipping.")
            continue

        if split_type == 'train':
            preprocess_for_bert(xml_file_path, domain, output_dir)
        else:
            prepare_test_data(xml_file_path, domain, output_dir)
    
    print("\n✨ BERT preprocessing complete! ✨")
