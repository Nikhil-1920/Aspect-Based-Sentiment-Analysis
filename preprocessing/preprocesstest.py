import sys
import os
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from joblib import load
import spacy
import nltk

# Add project root to Python path to allow for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Import project-specific modules ---
try:
    from preprocessing.preprocess import get_dependency_features, get_lexicon_features
    from models.utils.aspect_extractor import AspectExtractor
    print("✅ Successfully imported project modules.")
except ImportError as e:
    print(f"❌ Error: Could not import a required module: {e}")
    print("Please ensure your project structure is correct and you are running from the project root.")
    sys.exit()

# --- Initialize NLP models ---
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("❌ Spacy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    sys.exit()


def parse_test_xml(xml_file_path, domain, aspect_extractor):
    """Parse test XML and extract aspects for each sentence."""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    data = []
    
    for sentence in root.findall('sentence'):
        text_element = sentence.find('text')
        if text_element is None or text_element.text is None:
            continue
        text = text_element.text.replace('&quot;', '"').replace('&apos;', "'").strip()
        
        # Use the extractor to find aspects in the sentence
        aspects = aspect_extractor.extract_aspects(text, domain)
        
        for aspect in aspects:
            data.append({'sentence': text, 'aspect': aspect})
            
    return pd.DataFrame(data)

def create_test_features(df):
    """Create features for the test dataframe."""
    if 'sentence' not in df.columns or 'aspect' not in df.columns:
        print("⚠️ Warning: DataFrame is missing expected columns. Skipping feature creation.")
        return df

    df['text'] = df['sentence'] + " " + df['aspect']
    lexicon_features = df['sentence'].apply(get_lexicon_features).apply(pd.Series)
    dep_features = df.apply(lambda x: get_dependency_features(x['sentence'], x['aspect']), axis=1).apply(pd.Series)
    
    return df.join(lexicon_features).join(dep_features)


if __name__ == "__main__":
    # --- Define Paths ---
    raw_data_dir = 'data/raw'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    test_files = [
        ("Laptops_Test_Data_PhaseA.xml", "laptop"),
        ("Restaurants_Test_Data_PhaseA.xml", "restaurant")
    ]
    
    # Initialize the aspect extractor
    print("Initializing Aspect Extractor...")
    aspect_extractor = AspectExtractor()
    
    for test_xml, domain in test_files:
        title = f"Processing test data for: {domain.upper()}"
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}")
        
        raw_xml_path = os.path.join(raw_data_dir, test_xml)
        if not os.path.exists(raw_xml_path):
            print(f"❌ Error: Test file not found at '{raw_xml_path}'. Skipping.")
            continue
            
        # 1. Parse XML and extract aspects
        print(f"📝 Parsing '{test_xml}' and extracting aspects...")
        test_df = parse_test_xml(raw_xml_path, domain, aspect_extractor)
        if test_df.empty:
            print("...No aspects extracted. Skipping domain.")
            continue
        print(f"...Extracted {len(test_df)} sentence-aspect pairs.")

        # 2. Create features
        print("🔧 Creating features for test data...")
        test_df = create_test_features(test_df)
        
        try:
            # 3. Load trained vectorizers from the 'processed' directory
            print("🧠 Loading trained vectorizers...")
            tfidf = load(os.path.join(processed_dir, f"{domain}_tfidf_vectorizer.joblib"))
            bow = load(os.path.join(processed_dir, f"{domain}_bow_vectorizer.joblib"))

            # 4. Transform features using the loaded vectorizers
            print("...Applying transformations...")
            X_test_tfidf = tfidf.transform(test_df['text'])
            X_test_bow = bow.transform(test_df['text'])
            
            X_test = np.hstack([
                X_test_tfidf.toarray(),
                X_test_bow.toarray(),
                test_df[['lexicon_pos', 'lexicon_neg']].values
            ])
            
            # 5. Save the final unlabeled test data to the 'processed' directory
            output_csv_path = os.path.join(processed_dir, f"{domain}_test_features_unlabeled.csv")
            output_npy_path = os.path.join(processed_dir, f"{domain}_X_test_unlabeled.npy")
            
            print(f"💾 Saving processed test files for '{domain}'...")
            test_df.to_csv(output_csv_path, index=False)
            np.save(output_npy_path, X_test)
            print(f"...Successfully saved to '{output_csv_path}'")

        except FileNotFoundError as e:
            print(f"\n❌ FATAL ERROR: Could not find a required trained model file for '{domain}'.")
            print("Please ensure you have run the main `preprocess.py` script first to generate these files.")
            print(f"Missing file: {e.filename}")
        except Exception as e:
            print(f"An unexpected error occurred for domain '{domain}': {e}")

    print("\n✨ Test data processing complete! ✨")


